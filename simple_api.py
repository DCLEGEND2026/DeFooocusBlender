"""
Simple API Extension for DeFooocus
Add this to your DeFooocus installation to enable easy API access
"""

import gradio as gr
import modules.async_worker as worker
import modules.config
import shared

def create_simple_api():
    """Create a simplified API endpoint for external tools"""
    
    def simple_generate(
        prompt="a beautiful landscape",
        negative_prompt="",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.0,
        seed=-1,
        performance="Speed"
    ):
        """
        Simplified generation function for API use
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (default 1024)
            height: Image height (default 1024)
            steps: Number of sampling steps (default 30, use -1 for auto)
            cfg_scale: Guidance scale (default 7.0)
            seed: Random seed (-1 for random)
            performance: "Speed", "Quality", or "Extreme Speed"
        
        Returns:
            List of generated image paths
        """
        
        # Build aspect ratio string
        aspect_ratio = f"{width}×{height}"
        
        # Build the full args array matching webui.py structure
        # [currentTask is removed by get_task, so we start with generate_image_grid]
        args = [
            False,  # generate_image_grid
            # Main generation parameters
            prompt,
            negative_prompt,
            False,  # translate_prompts
            [],  # style_selections
            performance,
            aspect_ratio,
            1,  # image_number
            "png",  # output_format
            seed,
            cfg_scale,  # sharpness
            cfg_scale,  # guidance_scale
            # Model settings - use defaults from config
            modules.config.default_base_model_name,
            modules.config.default_refiner_model_name,
            modules.config.default_refiner_switch,
            # LoRA settings - use defaults
            *[item for lora in modules.config.default_loras for item in lora],
            # Input image settings - all disabled
            False,  # input_image_checkbox
            "uov",  # current_tab
            "Disabled",  # uov_method
            None,  # uov_input_image
            [],  # outpaint_selections
            None,  # inpaint_input_image
            "",  # inpaint_additional_prompt
            None,  # inpaint_mask_image
            # Preview settings
            False,  # disable_preview
            False,  # disable_intermediate_results
            modules.config.default_black_out_nsfw,
            # ADM settings - use defaults
            1.5,  # adm_scaler_positive
            0.8,  # adm_scaler_negative
            0.3,  # adm_scaler_end
            modules.config.default_cfg_tsnr,
            # Sampler settings
            modules.config.default_sampler,
            modules.config.default_scheduler,
            # Overwrite settings
            steps if steps > 0 else modules.config.default_overwrite_step,
            modules.config.default_overwrite_switch,
            -1,  # overwrite_width
            -1,  # overwrite_height
            -1,  # overwrite_vary_strength
            modules.config.default_overwrite_upscale,
            False,  # mixing_image_prompt_and_vary_upscale
            False,  # mixing_image_prompt_and_inpaint
            # Canny settings
            False,  # debugging_cn_preprocessor
            False,  # skipping_cn_preprocessor
            64,   # canny_low_threshold
            128,  # canny_high_threshold
            # Other settings
            "joint",  # refiner_swap_method
            0.25,  # controlnet_softness
            # FreeU settings
            False, 1.01, 1.02, 0.99, 0.95,
            # Inpaint settings
            False,  # debugging_inpaint_preprocessor
            False,  # inpaint_disable_initial_latent
            modules.config.default_inpaint_engine_version,
            1.0,  # inpaint_strength
            0.618,  # inpaint_respective_field
            False,  # inpaint_mask_upload_checkbox
            False,  # invert_mask_checkbox
            0,  # inpaint_erode_or_dilate
        ]
        
        # Add metadata settings if enabled
        if not hasattr(modules.config, 'default_save_metadata_to_images') or modules.config.default_save_metadata_to_images:
            args.extend([True, modules.config.default_metadata_scheme if hasattr(modules.config, 'default_metadata_scheme') else 'fooocus'])
        
        # Add image prompt settings (4 sets of: image, stop, weight, type)
        args.extend([None, 0.5, 0.6, "ImagePrompt"] * 4)
        
        # Create task
        task = worker.AsyncTask(args=args)
        
        # Wait for task to complete
        import time
        worker.async_tasks.append(task)
        
        # Poll for completion
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if len(task.yields) > 0:
                flag, product = task.yields[-1]  # Get last yield
                if flag == 'finish':
                    # Return the image paths
                    return product
            time.sleep(0.5)
        
        return ["Generation timed out"]
    
    # Create Gradio interface for the simple API
    simple_interface = gr.Interface(
        fn=simple_generate,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="a beautiful landscape"),
            gr.Textbox(label="Negative Prompt", value=""),
            gr.Number(label="Width", value=1024),
            gr.Number(label="Height", value=1024),
            gr.Number(label="Steps (-1 for auto)", value=30),
            gr.Number(label="CFG Scale", value=7.0),
            gr.Number(label="Seed (-1 for random)", value=-1),
            gr.Dropdown(label="Performance", choices=["Speed", "Quality", "Extreme Speed"], value="Speed"),
        ],
        outputs=gr.Gallery(label="Generated Images"),
        api_name="simple_generate",  # This creates /api/simple_generate endpoint
        title="DeFooocus Simple API",
        description="Simplified API for easy external tool integration"
    )
    
    return simple_interface

# This will be imported and mounted by webui.py
