bl_info = {
    "name": "DeFooocus Render Stylizer",
    "author": "DeFooocusBlender",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar (N) > DeFooocus",
    "description": "Send your render to a DeFooocus server and get it back in the style you choose",
    "category": "Render",
}

import base64
import http.cookiejar
import json
import os
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty


# Same names and creativity as STYLE_PRESETS in modules/blender_render.py; "Refresh Styles" pulls the live list.
DEFAULT_PRESETS = {
    'Photorealistic': 0.55, 'Cinematic': 0.6, 'Anime': 0.75, '3D Animated Movie': 0.65, 'Claymation': 0.7,
    'Watercolor': 0.75, 'Oil Painting': 0.75, 'Comic Book': 0.75, 'Cyberpunk Neon': 0.7, 'Fantasy Art': 0.7,
    'Pencil Sketch': 0.75, 'Pixel Art': 0.8, 'Low Poly': 0.7, 'Ukiyo-e': 0.8, 'Custom (prompt / styles only)': 0.6,
}
PERFORMANCES = ['Speed', 'Quality', 'Extreme Speed']
POLL_SECONDS = 1.0

# Blender keeps only a pointer to dynamic enum strings, so the item list must stay referenced here.
_style_items = [(name, name, '') for name in DEFAULT_PRESETS]
_preset_creativity = dict(DEFAULT_PRESETS)

_job = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': '',
    'results': [],
    'cancel': False,
    'preset': '',
}


# ---------------------------------------------------------------------------- server client

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class DeFooocusClient:
    def __init__(self, url, username='', password=''):
        self.url = url.strip().rstrip('/')
        self.username = username
        self.password = password
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
            urllib.request.HTTPSHandler(context=_ssl_context()))
        self.logged_in = False

    def login(self):
        if self.logged_in or not self.username:
            return
        form = urllib.parse.urlencode({'username': self.username, 'password': self.password}).encode()
        self.opener.open(urllib.request.Request(f'{self.url}/login', data=form), timeout=30)
        self.logged_in = True

    def call(self, api_name, *data, timeout=120):
        self.login()
        request = urllib.request.Request(
            f'{self.url}/run/{api_name}',
            data=json.dumps({'data': list(data)}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'Server error {e.code}: {e.read().decode(errors="replace")[:300]}')
        except urllib.error.URLError as e:
            raise RuntimeError(f'Cannot reach DeFooocus at {self.url} ({e.reason})')
        if payload.get('error'):
            raise RuntimeError(payload['error'])
        return json.loads(payload['data'][0])


def _client(context):
    prefs = context.preferences.addons[__name__].preferences
    return DeFooocusClient(prefs.server_url, prefs.username, prefs.password)


# ---------------------------------------------------------------------------- properties

def _style_enum_items(self, context):
    return _style_items


def _on_style_changed(self, context):
    if self.style in _preset_creativity:
        self.creativity = _preset_creativity[self.style]


class DeFooocusPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    server_url: StringProperty(
        name='Server URL', default='http://127.0.0.1:7865',
        description='DeFooocus address, e.g. http://127.0.0.1:7865 or the https://xxxx.gradio.live link from Colab')
    username: StringProperty(name='Username', description='Only if DeFooocus runs with auth.json')
    password: StringProperty(name='Password', subtype='PASSWORD')
    output_dir: StringProperty(
        name='Save Results To', subtype='DIR_PATH', default='//defooocus/',
        description='Folder for stylized images (// = next to the .blend file, temp folder if unsaved)')

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'server_url')
        row = layout.row()
        row.prop(self, 'username')
        row.prop(self, 'password')
        layout.prop(self, 'output_dir')


class DeFooocusSettings(bpy.types.PropertyGroup):
    source: EnumProperty(
        name='Source',
        items=[('RENDER', 'Render Result', 'The last F12 render'),
               ('ACTIVE', 'Image in Editor', 'The image currently shown in this Image Editor')],
        default='RENDER')
    style: EnumProperty(name='Style', items=_style_enum_items, update=_on_style_changed)
    prompt: StringProperty(name='Prompt', description='Describe the scene (optional), e.g. "a cozy cabin in a snowy forest"')
    negative_prompt: StringProperty(name='Negative', description='What to avoid (optional)')
    extra_styles: StringProperty(
        name='Extra Styles', description='Comma separated DeFooocus/SDXL style names, e.g. "Fooocus Enhance, sai-cinematic"')
    creativity: FloatProperty(
        name='Creativity', default=0.55, min=0.05, max=1.0,
        description='Low keeps the colors and lighting of the render, high repaints it more freely')
    structure_strength: FloatProperty(
        name='Structure Strength', default=1.0, min=0.0, max=2.0,
        description='How strictly the edges and shapes of the render are followed')
    structure_stop: FloatProperty(
        name='Structure Until', default=0.6, min=0.0, max=1.0,
        description='Fraction of the sampling steps that follows the structure')
    use_edges: BoolProperty(name='Edges', default=True, description='Follow the edges of the render (PyraCanny)')
    use_shapes: BoolProperty(name='Shapes', default=True, description='Follow the shapes and depth of the render (CPDS)')
    performance: EnumProperty(name='Performance', items=[(p, p, '') for p in PERFORMANCES], default='Speed')
    image_count: IntProperty(name='Images', default=1, min=1, max=4)
    seed: IntProperty(name='Seed', default=-1, min=-1, description='-1 for a random seed')
    match_size: BoolProperty(name='Match Render Size', default=True,
                             description='Scale the result to the exact resolution of the render')


# ---------------------------------------------------------------------------- helpers

def _redraw_image_editors():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def _show_in_image_editor(image):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = image
                return


def _output_dir():
    prefs = bpy.context.preferences.addons[__name__].preferences
    folder = prefs.output_dir or '//defooocus/'
    if folder.startswith('//') and not bpy.data.filepath:
        folder = os.path.join(tempfile.gettempdir(), 'defooocus')
    folder = bpy.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_source_png(context, source):
    if source == 'RENDER':
        image = bpy.data.images.get('Render Result')
        if image is None:
            raise RuntimeError('No render yet. Press F12 first (or use "Render + Stylize").')
    else:
        space = context.space_data
        image = getattr(space, 'image', None)
        if image is None:
            raise RuntimeError('No image is open in this Image Editor.')

    path = os.path.join(tempfile.gettempdir(), f'defooocus_source_{os.getpid()}.png')
    image_settings = context.scene.render.image_settings
    saved = (image_settings.file_format, image_settings.color_mode, image_settings.color_depth)
    try:
        # save_render() writes in the scene's output format, so force 8-bit PNG for the upload.
        image_settings.file_format = 'PNG'
        image_settings.color_mode = 'RGBA'
        image_settings.color_depth = '8'
        image.save_render(filepath=path, scene=context.scene)
    except RuntimeError as e:
        raise RuntimeError(f'Could not read "{image.name}": {e}. Render something first.')
    finally:
        image_settings.file_format, image_settings.color_mode, image_settings.color_depth = saved
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def _run_job(client, render_b64, params, out_dir):
    """Runs in a background thread: never touch bpy here."""
    try:
        _job['message'] = 'Uploading render ...'
        submitted = client.call('blender_submit', render_b64, json.dumps(params), timeout=300)
        if not submitted.get('ok'):
            raise RuntimeError(submitted.get('error', 'Submit failed'))
        job_id = submitted['job_id']

        stop_sent = False
        while True:
            if _job['cancel'] and not stop_sent:
                client.call('blender_stop', job_id)
                stop_sent = True
            status = client.call('blender_status', job_id, timeout=300)
            _job['progress'] = status.get('progress', 0)
            _job['message'] = status.get('message', '')
            if status['state'] not in ('queued', 'running'):
                break
            time.sleep(POLL_SECONDS)

        if status['state'] != 'done':
            raise RuntimeError('Stopped.' if stop_sent else status.get('message', 'Generation failed.'))

        stamp = time.strftime('%Y%m%d_%H%M%S')
        paths = []
        for i, image_b64 in enumerate(status['images']):
            path = os.path.join(out_dir, f'defooocus_{stamp}_{i + 1}.png')
            with open(path, 'wb') as f:
                f.write(base64.b64decode(image_b64))
            paths.append(path)
        _job['results'] = paths
        _job['message'] = f'Done: {len(paths)} image(s) saved to {out_dir}'
    except Exception as e:
        _job['error'] = str(e)
    finally:
        _job['running'] = False


def _poll_job():
    """Main-thread timer: loads finished images into Blender."""
    _redraw_image_editors()
    if _job['running']:
        return 0.5

    if _job['results']:
        first = None
        for i, path in enumerate(_job['results']):
            image = bpy.data.images.load(path, check_existing=False)
            image.name = f"DeFooocus {_job['preset']} {i + 1}"
            first = first or image
        _job['results'] = []
        if first is not None:
            _show_in_image_editor(first)
    return None


# ---------------------------------------------------------------------------- operators

class DEFOOOCUS_OT_refresh_styles(bpy.types.Operator):
    bl_idname = 'defooocus.refresh_styles'
    bl_label = 'Refresh Styles'
    bl_description = 'Check the server connection and load its style list'

    def execute(self, context):
        global _style_items
        try:
            info = _client(context).call('blender_styles', timeout=30)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        _style_items = [(p['name'], p['name'], ', '.join(p['styles'])) for p in info['presets']]
        _preset_creativity.clear()
        _preset_creativity.update({p['name']: p['creativity'] for p in info['presets']})
        self.report({'INFO'}, f'Connected: {len(_style_items)} style presets, {len(info["styles"])} extra styles')
        return {'FINISHED'}


class DEFOOOCUS_OT_stylize(bpy.types.Operator):
    bl_idname = 'defooocus.stylize'
    bl_label = 'Stylize'
    bl_description = 'Send the image to DeFooocus and load the stylized result'

    render_first: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        if _job['running']:
            self.report({'WARNING'}, 'A DeFooocus job is already running.')
            return {'CANCELLED'}
        settings = context.scene.defooocus

        source = settings.source
        if self.render_first:
            bpy.ops.render.render()
            source = 'RENDER'

        try:
            render_b64 = _save_source_png(context, source)
            out_dir = _output_dir()
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        params = dict(
            preset_name=settings.style,
            prompt=settings.prompt,
            negative_prompt=settings.negative_prompt,
            extra_styles=[s.strip() for s in settings.extra_styles.split(',') if s.strip()],
            creativity=settings.creativity,
            structure_weight=settings.structure_strength,
            structure_stop=settings.structure_stop,
            use_edges=settings.use_edges,
            use_shapes=settings.use_shapes,
            performance=settings.performance,
            image_number=settings.image_count,
            seed=settings.seed,
            match_size=settings.match_size,
        )

        _job.update(running=True, progress=0, message='Starting ...', error='', results=[], cancel=False,
                    preset=settings.style)
        threading.Thread(target=_run_job, args=(_client(context), render_b64, params, out_dir), daemon=True).start()
        bpy.app.timers.register(_poll_job, first_interval=0.5)
        return {'FINISHED'}


class DEFOOOCUS_OT_stop(bpy.types.Operator):
    bl_idname = 'defooocus.stop'
    bl_label = 'Stop'
    bl_description = 'Stop the running DeFooocus job'

    def execute(self, context):
        _job['cancel'] = True
        return {'FINISHED'}


# ---------------------------------------------------------------------------- UI

class DEFOOOCUS_PT_panel(bpy.types.Panel):
    bl_label = 'DeFooocus Stylizer'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'DeFooocus'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.defooocus
        prefs = context.preferences.addons[__name__].preferences

        row = layout.row(align=True)
        row.label(text=prefs.server_url, icon='WORLD')
        row.operator('defooocus.refresh_styles', text='', icon='FILE_REFRESH')

        layout.prop(settings, 'source')
        layout.prop(settings, 'style')
        layout.prop(settings, 'prompt')
        layout.prop(settings, 'creativity', slider=True)

        box = layout.box()
        box.label(text='Structure', icon='MESH_DATA')
        row = box.row(align=True)
        row.prop(settings, 'use_edges', toggle=True)
        row.prop(settings, 'use_shapes', toggle=True)
        box.prop(settings, 'structure_strength', slider=True)
        box.prop(settings, 'structure_stop', slider=True)

        box = layout.box()
        box.label(text='Options', icon='PREFERENCES')
        box.prop(settings, 'negative_prompt')
        box.prop(settings, 'extra_styles')
        box.prop(settings, 'performance')
        row = box.row(align=True)
        row.prop(settings, 'image_count')
        row.prop(settings, 'seed')
        box.prop(settings, 'match_size')

        if _job['running']:
            if hasattr(layout, 'progress'):
                layout.progress(factor=_job['progress'] / 100.0, text=_job['message'])
            else:
                layout.label(text=f"{_job['progress']}%  {_job['message']}", icon='TIME')
            layout.operator('defooocus.stop', icon='CANCEL')
        else:
            col = layout.column(align=True)
            col.scale_y = 1.4
            col.operator('defooocus.stylize', text='Stylize', icon='BRUSH_DATA')
            col.operator('defooocus.stylize', text='Render + Stylize', icon='RENDER_STILL').render_first = True
            if _job['error']:
                layout.label(text=_job['error'], icon='ERROR')
            elif _job['message']:
                layout.label(text=_job['message'], icon='CHECKMARK')


classes = (
    DeFooocusPreferences,
    DeFooocusSettings,
    DEFOOOCUS_OT_refresh_styles,
    DEFOOOCUS_OT_stylize,
    DEFOOOCUS_OT_stop,
    DEFOOOCUS_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.defooocus = PointerProperty(type=DeFooocusSettings)


def unregister():
    if bpy.app.timers.is_registered(_poll_job):
        bpy.app.timers.unregister(_poll_job)
    del bpy.types.Scene.defooocus
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
