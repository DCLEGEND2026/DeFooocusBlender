"""
Blender Render -> Stylized Image

Turns a Blender render into a stylized image while keeping its composition.
It mixes three things in a single DeFooocus task:

  * "Vary" on the render itself (keeps colors / lighting, amount controlled by `creativity`)
  * PyraCanny + CPDS ControlNets on the render (keeps edges / shapes / perspective)
  * SDXL styles + prompt from the chosen style preset (gives the look)

Used by the "Blender Render" tab in webui.py and by the HTTP API that the
Blender add-on (blender_addon/defooocus_blender.py) talks to.
"""

import base64
import io
import json
import os
import random
import threading
import time
import uuid

import numpy as np
from PIL import Image

import args_manager
import modules.config
import modules.constants as constants
import modules.flags as flags
import modules.async_worker as worker
from modules.sdxl_styles import legal_style_names, normalize_key


# name -> styles, prompt, negative prompt, default creativity (vary denoise strength)
STYLE_PRESETS = {
    'Photorealistic': dict(
        styles=['Fooocus Photograph', 'Fooocus Enhance', 'Fooocus Sharp'],
        prompt='photorealistic, real photo, natural lighting, realistic materials, highly detailed textures',
        negative='3d render, cgi, cartoon, plastic, painting, illustration',
        creativity=0.55),
    'Cinematic': dict(
        styles=['Fooocus Cinematic', 'Fooocus Enhance', 'Fooocus Sharp'],
        prompt='cinematic film still, dramatic lighting, shallow depth of field, color graded',
        negative='cartoon, illustration, flat lighting',
        creativity=0.6),
    'Anime': dict(
        styles=['sai-anime', 'mre-anime'],
        prompt='anime style, cel shading, clean line art, vibrant colors',
        negative='photo, realistic, 3d render',
        creativity=0.75),
    '3D Animated Movie': dict(
        styles=['sai-3d-model', 'Adorable 3D Character'],
        prompt='stylized 3d animated movie still, soft global illumination, appealing shapes, subsurface scattering',
        negative='photo, realistic, gritty, low quality',
        creativity=0.65),
    'Claymation': dict(
        styles=['sai-craft clay'],
        prompt='claymation, plasticine, handmade clay figures, stop motion',
        negative='photo, realistic, smooth cgi',
        creativity=0.7),
    'Watercolor': dict(
        styles=['artstyle-watercolor', 'Watercolor 2'],
        prompt='watercolor painting, soft washes, paper texture, loose brush strokes',
        negative='photo, 3d render, sharp digital',
        creativity=0.75),
    'Oil Painting': dict(
        styles=['artstyle-impressionist', 'Fooocus Masterpiece'],
        prompt='oil painting on canvas, visible thick brush strokes, rich colors',
        negative='photo, 3d render, digital art',
        creativity=0.75),
    'Comic Book': dict(
        styles=['sai-comic book', 'mre-comic'],
        prompt='comic book art, bold ink outlines, halftone shading',
        negative='photo, realistic, 3d render',
        creativity=0.75),
    'Cyberpunk Neon': dict(
        styles=['sai-neonpunk', 'Fooocus Enhance'],
        prompt='cyberpunk, neon lights, night, glowing signs, wet reflective surfaces',
        negative='daylight, dull colors',
        creativity=0.7),
    'Fantasy Art': dict(
        styles=['sai-fantasy art', 'Fooocus Masterpiece'],
        prompt='epic fantasy concept art, magical atmosphere, painterly',
        negative='photo, modern',
        creativity=0.7),
    'Pencil Sketch': dict(
        styles=['Pencil Sketch Drawing'],
        prompt='pencil sketch, graphite, hand drawn, cross hatching',
        negative='color, photo, 3d render',
        creativity=0.75),
    'Pixel Art': dict(
        styles=['sai-pixel art'],
        prompt='pixel art, 16-bit game art, limited palette',
        negative='photo, blurry, smooth gradients',
        creativity=0.8),
    'Low Poly': dict(
        styles=['sai-lowpoly'],
        prompt='low poly art, flat shaded polygons, geometric',
        negative='photo, high detail textures',
        creativity=0.7),
    'Ukiyo-e': dict(
        styles=['MK Ukiyo-e'],
        prompt='ukiyo-e japanese woodblock print, flat colors, bold outlines',
        negative='photo, 3d render',
        creativity=0.8),
    'Custom (prompt / styles only)': dict(
        styles=[],
        prompt='',
        negative='',
        creativity=0.6),
}

DEFAULT_PRESET = 'Photorealistic'

# Style names are normalized on load ('sai-anime' -> 'SAI Anime'). Drop any that do not exist in this
# install, so a renamed style never breaks a preset.
for _name, _preset in STYLE_PRESETS.items():
    _styles = [normalize_key(s) for s in _preset['styles']]
    for _missing in [s for s in _styles if s not in legal_style_names]:
        print(f'[Blender Render] Style "{_missing}" of preset "{_name}" not found, skipping it.')
    _preset['styles'] = [s for s in _styles if s in legal_style_names]

# (stop at, weight) used for the structure ControlNets when nothing else is given
DEFAULT_STRUCTURE_STOP = 0.6
DEFAULT_STRUCTURE_WEIGHT = 1.0
STYLE_REFERENCE_STOP, STYLE_REFERENCE_WEIGHT = flags.default_parameters[flags.cn_ip]

MAX_IMAGES = 4
JOB_TTL_SECONDS = 60 * 60

jobs = {}
jobs_lock = threading.Lock()


def preset_names():
    return list(STYLE_PRESETS.keys())


def preset_creativity(preset_name):
    return STYLE_PRESETS.get(preset_name, STYLE_PRESETS[DEFAULT_PRESET])['creativity']


def to_rgb_array(image):
    """Accepts a numpy array, PIL image, or base64 / data-URL string. Returns HxWx3 uint8."""
    if image is None or (isinstance(image, str) and image.strip() == ''):
        return None
    if isinstance(image, str):
        if image.startswith('data:'):
            image = image.split(',', 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(image)))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA', 'P'):
        # Blender "Film > Transparent" renders: put them on white, same as HWC3 does.
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return np.array(image.convert('RGB'), dtype=np.uint8)


def build_prompts(preset_name, prompt, negative_prompt, extra_styles):
    preset = STYLE_PRESETS.get(preset_name, STYLE_PRESETS[DEFAULT_PRESET])
    positive = ', '.join(p for p in [(prompt or '').strip(), preset['prompt']] if p)
    negative = ', '.join(p for p in [(negative_prompt or '').strip(), preset['negative']] if p)
    styles = list(preset['styles'])
    for s in extra_styles or []:
        s = normalize_key(s.strip())
        if s in legal_style_names and s not in styles:
            styles.append(s)
    return positive, negative, styles


def build_task_args(render, positive, negative, styles,
                    creativity=0.6,
                    structure_weight=DEFAULT_STRUCTURE_WEIGHT,
                    structure_stop=DEFAULT_STRUCTURE_STOP,
                    use_edges=True,
                    use_shapes=True,
                    style_reference=None,
                    performance=None,
                    image_number=1,
                    seed=-1,
                    guidance_scale=None,
                    sharpness=None):
    """
    Builds the argument list in exactly the order the "ctrls" list in webui.py sends to
    modules.async_worker (minus currentTask, which get_task() strips).
    """
    if performance not in flags.Performance.list():
        performance = modules.config.default_performance
    seed = int(seed)
    if seed < constants.MIN_SEED or seed > constants.MAX_SEED:
        seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    creativity = float(np.clip(creativity, 0.05, 1.0))
    image_number = int(np.clip(int(image_number), 1, MAX_IMAGES))

    args = [
        False,  # generate_image_grid
        positive,
        negative,
        False,  # translate_prompts
        styles,
        performance,
        '1024×1024',  # aspect_ratios_selection, ignored: "Vary" takes the size from the render
        image_number,
        'png',
        seed,
        modules.config.default_sample_sharpness if sharpness is None else float(sharpness),
        modules.config.default_cfg_scale if guidance_scale is None else float(guidance_scale),
        modules.config.default_base_model_name,
        modules.config.default_refiner_model_name,
        modules.config.default_refiner_switch,
    ]
    for lora_name, lora_weight in modules.config.default_loras:
        args += [lora_name, lora_weight]

    args += [True, 'ip']  # input_image_checkbox, current_tab
    args += [flags.strong_variation, render]  # uov_method, uov_input_image
    args += [[], None, '', None]  # outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image
    args += [False, performance == flags.Performance.EXTREME_SPEED.value, modules.config.default_black_out_nsfw]
    args += [1.5, 0.8, 0.3, modules.config.default_cfg_tsnr]  # adm scalers, adaptive_cfg
    args += [modules.config.default_sampler, modules.config.default_scheduler]
    args += [modules.config.default_overwrite_step, modules.config.default_overwrite_switch, -1, -1, creativity]
    args += [modules.config.default_overwrite_upscale, True, False]  # mixing_image_prompt_and_vary_upscale = True
    args += [False, False, 64, 128]  # debugging_cn_preprocessor, skipping_cn_preprocessor, canny thresholds
    args += [flags.refiner_swap_method, 0.25]  # refiner_swap_method, controlnet_softness
    args += [False, 1.01, 1.02, 0.99, 0.95]  # freeu
    args += [False, False, modules.config.default_inpaint_engine_version, 1.0, 0.618, False, False, 0]  # inpaint

    if not args_manager.args.disable_metadata:
        args += [modules.config.default_save_metadata_to_images, modules.config.default_metadata_scheme]

    image_prompts = []
    if use_edges:
        image_prompts.append([render, structure_stop, structure_weight, flags.cn_canny])
    if use_shapes:
        image_prompts.append([render, structure_stop, structure_weight, flags.cn_cpds])
    if style_reference is not None:
        image_prompts.append([style_reference, STYLE_REFERENCE_STOP, STYLE_REFERENCE_WEIGHT, flags.cn_ip])
    while len(image_prompts) < flags.controlnet_image_count:
        image_prompts.append([None, 0.5, 0.6, flags.cn_ip])
    for image_prompt in image_prompts:
        args += image_prompt

    return args


def cleanup_jobs():
    now = time.time()
    with jobs_lock:
        for job_id in [k for k, v in jobs.items() if now - v['created'] > JOB_TTL_SECONDS]:
            del jobs[job_id]


def submit(render, preset_name=DEFAULT_PRESET, prompt='', negative_prompt='', extra_styles=None,
           creativity=None, structure_weight=DEFAULT_STRUCTURE_WEIGHT, structure_stop=DEFAULT_STRUCTURE_STOP,
           use_edges=True, use_shapes=True, style_reference=None, performance=None, image_number=1, seed=-1,
           guidance_scale=None, sharpness=None, match_size=False):
    """Queues a stylize job on the DeFooocus worker. Returns the job id."""
    import ldm_patched.modules.model_management as model_management

    render = to_rgb_array(render)
    if render is None:
        raise ValueError('No render image given.')
    style_reference = to_rgb_array(style_reference)
    if creativity is None:
        creativity = preset_creativity(preset_name)

    positive, negative, styles = build_prompts(preset_name, prompt, negative_prompt, extra_styles)
    args = build_task_args(render, positive, negative, styles,
                           creativity=creativity,
                           structure_weight=structure_weight,
                           structure_stop=structure_stop,
                           use_edges=use_edges,
                           use_shapes=use_shapes,
                           style_reference=style_reference,
                           performance=performance,
                           image_number=image_number,
                           seed=seed,
                           guidance_scale=guidance_scale,
                           sharpness=sharpness)

    cleanup_jobs()
    task = worker.AsyncTask(args=args)
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = dict(task=task, created=time.time(),
                            size=(render.shape[1], render.shape[0]) if match_size else None)

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    worker.async_tasks.append(task)
    print(f'[Blender Render] Job {job_id} queued: preset={preset_name}, styles={styles}, creativity={creativity}')
    return job_id


def result_images(job_id):
    """Returns the finished images of a job as PIL images (resized to the render size if asked)."""
    job = jobs[job_id]
    images = []
    for path in job['task'].results:
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        image = Image.open(path).convert('RGB')
        if job['size'] is not None and image.size != job['size']:
            image = image.resize(job['size'], Image.LANCZOS)
        images.append(image)
    return images


def status(job_id, include_images=True):
    """
    state is one of: queued, running, done, failed, unknown
    progress is 0-100, images are base64 PNGs (only when done)
    """
    job = jobs.get(job_id)
    if job is None:
        return dict(state='unknown', progress=0, message='Unknown job id (server restarted?)', images=[])

    task = job['task']
    if len(task.yields) == 0:
        try:
            position = worker.async_tasks.index(task)
            message = f'Waiting in queue (position {position + 1}) ...'
        except ValueError:
            message = 'Starting ...'
        return dict(state='queued', progress=0, message=message, images=[])

    finished = any(flag == 'finish' for flag, _ in task.yields)
    if not finished:
        progress, message = 0, 'Running ...'
        for flag, product in reversed(task.yields):
            if flag == 'preview':
                progress, message = int(product[0]), str(product[1])
                break
        return dict(state='running', progress=progress, message=message, images=[])

    images = result_images(job_id)
    if len(images) == 0:
        state = 'failed' if task.last_stop != 'stop' else 'stopped'
        return dict(state=state, progress=100, message='No image was produced, see the DeFooocus console.', images=[])

    encoded = []
    if include_images:
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    return dict(state='done', progress=100, message=f'Finished {len(images)} image(s).', images=encoded,
                paths=[p for p in task.results if isinstance(p, str)])


def stop(job_id):
    import ldm_patched.modules.model_management as model_management

    job = jobs.get(job_id)
    if job is None:
        return False
    task = job['task']
    task.last_stop = 'stop'
    try:
        worker.async_tasks.remove(task)
        task.yields.append(['finish', []])
    except ValueError:
        # Already picked up by the worker.
        if task.processing:
            model_management.interrupt_current_processing()
    return True


# ---- JSON API used by the Blender add-on (strings in, JSON strings out) ----

def api_styles():
    return json.dumps(dict(
        presets=[dict(name=k, creativity=v['creativity'], styles=v['styles']) for k, v in STYLE_PRESETS.items()],
        styles=legal_style_names,
        performances=flags.Performance.list(),
        default_preset=DEFAULT_PRESET))


def api_submit(render_b64, params_json):
    try:
        params = json.loads(params_json) if params_json else {}
        allowed = submit.__code__.co_varnames[1:submit.__code__.co_argcount]
        params = {k: v for k, v in params.items() if k in allowed}
        job_id = submit(render_b64, **params)
        return json.dumps(dict(ok=True, job_id=job_id))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps(dict(ok=False, error=str(e)))


def api_status(job_id):
    return json.dumps(status((job_id or '').strip()))


def api_stop(job_id):
    return json.dumps(dict(ok=stop((job_id or '').strip())))
