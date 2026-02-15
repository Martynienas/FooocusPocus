import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path, get_available_logs, get_latest_log
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        # Left panel for prompts
        with gr.Column(scale=2, min_width=300):
            gr.HTML("<h3>Prompts</h3>")
            prompt = gr.Textbox(label='Positive Prompt', placeholder="Type prompt here or paste parameters.", 
                                elem_id='positive_prompt', autofocus=True, lines=3, container=False)

            default_prompt = modules.config.default_prompt
            if isinstance(default_prompt, str) and default_prompt != '':
                shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

            negative_prompt = gr.Textbox(label='Negative Prompt', placeholder="Describing what you do not want to see.",
                                         lines=3, elem_id='negative_prompt', container=False,
                                         value=modules.config.default_prompt_negative)
            
            # Generate button
            generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
            
            # Control buttons row
            with gr.Row():
                reset_button = gr.Button(label="Reconnect", value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
            
            with gr.Row():
                skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

            def stop_clicked(currentTask):
                import ldm_patched.modules.model_management as model_management
                currentTask.last_stop = 'stop'
                if (currentTask.processing):
                    model_management.interrupt_current_processing()
                return currentTask

            def skip_clicked(currentTask):
                import ldm_patched.modules.model_management as model_management
                currentTask.last_stop = 'skip'
                if (currentTask.processing):
                    model_management.interrupt_current_processing()
                return currentTask

            stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
            skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            
            # Control checkboxes
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
        
        # Center panel for gallery and controls
        with gr.Column(scale=3):
            with gr.Row():
                progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                              height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')

            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=dino_prompt_text,
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)

                    with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='Content Type',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='Describe this Image into Prompt')
                                describe_image_size = gr.Textbox(label='Image Size and Recommended Size', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                                def trigger_show_image_properties(image):
                                    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                    return gr.update(value=value, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='pil')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='Upscale or Variation'):
                        with gr.Row():
                            with gr.Column():
                                enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                    enhance_ctrls = []
                    enhance_inpaint_mode_ctrls = []
                    enhance_inpaint_engine_ctrls = []
                    enhance_inpaint_update_ctrls = []
                    for index in range(modules.config.default_enhance_tabs):
                        with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                            enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check',
                                                          container=False)

                            enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text = gr.Dataset(
                                samples=modules.config.example_enhance_detection_prompts,
                                label='Detection Prompt Quick List',
                                components=[enhance_mask_dino_prompt_text],
                                visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                            enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                            enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                            with gr.Accordion("Detection", open=False):
                                enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                with gr.Accordion("SAM Options",
                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                  open=False) as sam_options:
                                    enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                    enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                    enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                    enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                            with gr.Accordion("Inpaint", visible=True, open=False):
                                enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method,
                                                                   label='Method', interactive=True)
                                enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                    label='Disable initial latent in inpaint', value=False)
                                enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                            gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                        enhance_ctrls += [
                            enhance_enabled,
                            enhance_mask_dino_prompt_text,
                            enhance_prompt,
                            enhance_negative_prompt,
                            enhance_mask_model,
                            enhance_mask_cloth_category,
                            enhance_mask_sam_model,
                            enhance_mask_text_threshold,
                            enhance_mask_box_threshold,
                            enhance_mask_sam_max_detections,
                            enhance_inpaint_disable_initial_latent,
                            enhance_inpaint_engine,
                            enhance_inpaint_strength,
                            enhance_inpaint_respective_field,
                            enhance_inpaint_erode_or_dilate,
                            enhance_mask_invert
                        ]

                        enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                        enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                        enhance_inpaint_update_ctrls += [[
                            enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ]]

                        enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                            enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ], show_progress=False, queue=False)

                        enhance_mask_model.change(
                            lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                      [gr.update(visible=x == 'sam')] * 2 +
                                      [gr.Dataset.update(visible=x == 'sam',
                                                         samples=modules.config.example_enhance_detection_prompts)],
                            inputs=enhance_mask_model,
                            outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                     example_enhance_mask_dino_prompt_text],
                            queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                        outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

        with gr.Column(scale=2, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Settings'):
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Dropdown(label='Preset',
                                                   choices=modules.config.available_presets,
                                                   value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                   interactive=True)

                with gr.Row():
                    steps_slider = gr.Slider(label='Steps', minimum=1, maximum=200, step=1, value=modules.config.default_steps,
                                           info='Number of sampling steps for image generation')
                    upscale_steps_slider = gr.Slider(label='Upscale Steps', minimum=1, maximum=200, step=1, value=modules.config.default_upscale_steps,
                                                   info='Number of steps for upscaling operations')

                with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                    aspect_ratios_selection = gr.Radio(label='Aspect Ratios', show_label=False,
                                                       choices=modules.config.available_aspect_ratios_labels,
                                                       value=modules.config.default_aspect_ratio,
                                                       info='width × height',
                                                       elem_classes='aspect_ratios')

                    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                # negative_prompt moved to main prompt area under the positive prompt
                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    # Try to find latest existing log.html; fall back to the generated current path
                    latest = get_latest_log()
                    if latest is None:
                        # use generated (may be non-existent) path so behavior is preserved
                        href = get_current_html_path(output_format)
                    else:
                        href = latest

                    return gr.update(value=f'<a href="file={href}" target="_blank">\U0001F4DA History Log</a>')

                # history link (we'll put it between the navigation arrows)
                history_link = gr.HTML()
                # state to hold available logs and currently selected index
                history_logs = gr.State([])
                history_index = gr.State(0)

                def load_history_state():
                    if args_manager.args.disable_image_log:
                        return gr.update(), [] , 0
                    logs = get_available_logs()
                    idx = len(logs) - 1 if len(logs) > 0 else 0
                    # return HTML value, logs list and index
                    href = logs[idx] if logs else get_current_html_path(output_format)
                    return gr.update(value=f'<a href="file={href}" target="_blank">\U0001F4DA History Log</a>'), logs, idx

                # Navigation buttons to move between existing logs; put the link (including date) between arrows
                with gr.Row():
                    prev_btn = gr.Button(value='<<', variant='secondary')
                    # history_link will be placed here (between the arrows)
                    with gr.Column(scale=1, min_width=200):
                        pass
                    next_btn = gr.Button(value='>>', variant='secondary')

                # place the history_link into the UI by re-creating the row contents in markup
                # NOTE: gradio layout will render components in the order they were created; we already created prev_btn and next_btn
                # We'll now insert the history_link between them by creating it earlier; to keep code simple we'll rely on the
                # existing history_link component created above and the layout will flow: prev_btn, history_link, next_btn.

                def load_history_state_full():
                    # returns: link_html, logs list, index
                    if args_manager.args.disable_image_log:
                        return gr.update(value=''), [], 0
                    logs = get_available_logs()
                    if not logs:
                        # no existing logs -> show no logs message and empty link
                        return gr.update(value='<small>No logs</small>'), [], 0
                    idx = len(logs) - 1
                    href = logs[idx]
                    label = os.path.basename(os.path.dirname(href))
                    # put the label into the anchor text so it appears as: "<< History Log - YYYY-MM-DD >>"
                    return gr.update(value=f'<a href="file={href}" target="_blank">\U0001F4DA History Log - {label}</a>'), logs, idx

                shared.gradio_root.load(load_history_state_full, outputs=[history_link, history_logs, history_index], queue=False, show_progress=False)

                def navigate_logs(direction, logs, idx):
                    # direction: -1 for prev, +1 for next
                    if not isinstance(logs, list) or len(logs) == 0:
                        return gr.update(value='<small>No logs</small>'), [], 0
                    idx = int(idx) if idx is not None else (len(logs) - 1)
                    idx = max(0, min(len(logs) - 1, idx + direction))
                    href = logs[idx]
                    label = os.path.basename(os.path.dirname(href))
                    return gr.update(value=f'<a href="file={href}" target="_blank">\U0001F4DA History Log - {label}</a>'), logs, idx

                # wire the navigation buttons; they update link, logs state and index state
                prev_btn.click(lambda logs, idx: navigate_logs(-1, logs, idx), inputs=[history_logs, history_index], outputs=[history_link, history_logs, history_index], queue=False, show_progress=False)
                next_btn.click(lambda logs, idx: navigate_logs(1, logs, idx), inputs=[history_logs, history_index], outputs=[history_link, history_logs, history_index], queue=False, show_progress=False)

            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

            with gr.Tab(label='Models'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None', modules.config.random_lora_name] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Row():
                    refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')
                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='Set as negative number to disable. For developer debugging.')

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=False,
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)

                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                       info='Show enhance masks in preview and final results')
                        debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')

                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked():
                    modules.config.update_files()
                    results = [gr.update(choices=modules.config.model_filenames)]
                    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None', modules.config.random_lora_name] + modules.config.lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

            # =========================================================================
            # Configuration Tab - Manage application configuration
            # =========================================================================
            with gr.Tab(label='Configuration'):
                # --- Save/Restore Buttons at TOP (always visible) ---
                with gr.Row(equal_height=True):
                    save_config_btn = gr.Button('💾 Save Configuration', variant='primary', scale=2, size='lg')
                    restore_all_btn = gr.Button('↺ Restore All to Defaults', variant='secondary', scale=1, size='lg')
                config_status = gr.HTML('')
                gr.HTML('<hr style="margin: 10px 0;">')
                
                # Store for config UI elements
                config_ui_elements = {}
                
                # --- Model Folders Section ---
                with gr.Accordion(label='📁 Model Folders', open=True, elem_classes=['config-section']):
                    gr.HTML('<p style="color: var(--body-text-color-subdued); font-size: 0.9em;">Add or remove folders where models are stored. Changes take effect immediately.</p>')
                    
                    # Checkpoints folders
                    with gr.Group():
                        gr.HTML('<label class="config-label">Checkpoint Folders</label>')
                        checkpoint_folders_state = gr.State(value=lambda: modules.config.get_model_folders('path_checkpoints'))
                        checkpoint_folders_display = gr.Dataframe(
                            headers=['Path'],
                            datatype=['str'],
                            col_count=(1, 'fixed'),
                            row_count=1,
                            value=lambda: [[p] for p in modules.config.get_model_folders('path_checkpoints')],
                            interactive=False,
                            elem_classes=['folder-display']
                        )
                        with gr.Row():
                            checkpoint_folder_input = gr.Textbox(label='Add Checkpoint Folder', placeholder='/path/to/checkpoints', scale=4)
                            checkpoint_add_btn = gr.Button('Add', variant='secondary', scale=1, elem_classes=['folder-btn'])
                            checkpoint_reset_btn = gr.Button('↺ Reset', variant='secondary', scale=1, elem_classes=['reset-btn'])
                    
                    # LoRA folders
                    with gr.Group():
                        gr.HTML('<label class="config-label">LoRA Folders</label>')
                        lora_folders_display = gr.Dataframe(
                            headers=['Path'],
                            datatype=['str'],
                            col_count=(1, 'fixed'),
                            row_count=1,
                            value=lambda: [[p] for p in modules.config.get_model_folders('path_loras')],
                            interactive=False,
                            elem_classes=['folder-display']
                        )
                        with gr.Row():
                            lora_folder_input = gr.Textbox(label='Add LoRA Folder', placeholder='/path/to/loras', scale=4)
                            lora_add_btn = gr.Button('Add', variant='secondary', scale=1, elem_classes=['folder-btn'])
                            lora_reset_btn = gr.Button('↺ Reset', variant='secondary', scale=1, elem_classes=['reset-btn'])
                    
                    # Other model paths (single folder)
                    with gr.Group():
                        gr.HTML('<label class="config-label">Other Model Paths</label>')
                        with gr.Row():
                            embeddings_path = gr.Textbox(label='Embeddings', value=modules.config.path_embeddings, scale=3, interactive=True)
                            embeddings_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                        with gr.Row():
                            vae_path = gr.Textbox(label='VAE', value=modules.config.path_vae, scale=3, interactive=True)
                            vae_path_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                        with gr.Row():
                            controlnet_path = gr.Textbox(label='ControlNet', value=modules.config.path_controlnet, scale=3, interactive=True)
                            controlnet_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                        with gr.Row():
                            upscale_path = gr.Textbox(label='Upscale Models', value=modules.config.path_upscale_models, scale=3, interactive=True)
                            upscale_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                
                # --- Output & Paths Section ---
                with gr.Accordion(label='📂 Output & Paths', open=False, elem_classes=['config-section']):
                    with gr.Row():
                        output_path = gr.Textbox(label='Output Path', value=modules.config.path_outputs, scale=3, interactive=True)
                        output_path_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        temp_path_config = gr.Textbox(label='Temp Path', value=modules.config.temp_path, scale=3, interactive=True)
                        temp_path_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        temp_cleanup = gr.Checkbox(label='Cleanup Temp on Launch', value=modules.config.temp_path_cleanup_on_launch, interactive=True)
                        temp_cleanup_reset = gr.Button('↺ Reset', variant='secondary', elem_classes=['reset-btn'])
                
                # --- Default Models Section ---
                with gr.Accordion(label='🎨 Default Models', open=False, elem_classes=['config-section']):
                    with gr.Row():
                        config_default_model = gr.Dropdown(
                            label='Default Base Model',
                            choices=modules.config.model_filenames,
                            value=modules.config.default_base_model_name,
                            scale=3,
                            interactive=True
                        )
                        default_model_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_default_refiner = gr.Dropdown(
                            label='Default Refiner Model',
                            choices=['None'] + modules.config.model_filenames,
                            value=modules.config.default_refiner_model_name,
                            scale=3,
                            interactive=True
                        )
                        default_refiner_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_refiner_switch = gr.Slider(
                            label='Default Refiner Switch',
                            minimum=0.1, maximum=1.0, step=0.01,
                            value=modules.config.default_refiner_switch,
                            scale=3,
                            interactive=True
                        )
                        refiner_switch_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_default_vae = gr.Dropdown(
                            label='Default VAE',
                            choices=[flags.default_vae] + modules.config.vae_filenames,
                            value=modules.config.default_vae,
                            scale=3,
                            interactive=True
                        )
                        default_vae_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                
                # --- Default Generation Settings Section ---
                with gr.Accordion(label='⚡ Default Generation Settings', open=False, elem_classes=['config-section']):
                    with gr.Row():
                        config_default_steps = gr.Slider(
                            label='Default Steps',
                            minimum=1, maximum=200, step=1,
                            value=modules.config.default_steps,
                            scale=3,
                            interactive=True
                        )
                        default_steps_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_upscale_steps = gr.Slider(
                            label='Default Upscale Steps',
                            minimum=1, maximum=200, step=1,
                            value=modules.config.default_upscale_steps,
                            scale=3,
                            interactive=True
                        )
                        upscale_steps_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_cfg_scale = gr.Slider(
                            label='Default CFG Scale',
                            minimum=1.0, maximum=30.0, step=0.1,
                            value=modules.config.default_cfg_scale,
                            scale=3,
                            interactive=True
                        )
                        cfg_scale_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_sharpness = gr.Slider(
                            label='Default Sharpness',
                            minimum=0.0, maximum=30.0, step=0.01,
                            value=modules.config.default_sample_sharpness,
                            scale=3,
                            interactive=True
                        )
                        sharpness_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_sampler = gr.Dropdown(
                            label='Default Sampler',
                            choices=flags.sampler_list,
                            value=modules.config.default_sampler,
                            scale=3,
                            interactive=True
                        )
                        sampler_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_scheduler = gr.Dropdown(
                            label='Default Scheduler',
                            choices=flags.scheduler_list,
                            value=modules.config.default_scheduler,
                            scale=3,
                            interactive=True
                        )
                        scheduler_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_clip_skip = gr.Slider(
                            label='Default CLIP Skip',
                            minimum=1, maximum=flags.clip_skip_max, step=1,
                            value=modules.config.default_clip_skip,
                            scale=3,
                            interactive=True
                        )
                        clip_skip_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_adaptive_cfg = gr.Slider(
                            label='Default Adaptive CFG',
                            minimum=1.0, maximum=30.0, step=0.1,
                            value=modules.config.default_cfg_tsnr,
                            scale=3,
                            interactive=True
                        )
                        adaptive_cfg_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                
                # --- Default Styles Section ---
                with gr.Accordion(label='🎭 Default Styles', open=False, elem_classes=['config-section']):
                    config_default_styles = gr.CheckboxGroup(
                        label='Default Styles',
                        choices=modules.sdxl_styles.legal_style_names,
                        value=modules.config.default_styles,
                        elem_classes=['config-styles'],
                        interactive=True
                    )
                    default_styles_reset = gr.Button('↺ Reset to Default Styles', variant='secondary', elem_classes=['reset-btn'])
                
                # --- Image Settings Section ---
                with gr.Accordion(label='🖼️ Image Settings', open=False, elem_classes=['config-section']):
                    with gr.Row():
                        config_image_number = gr.Slider(
                            label='Default Image Number',
                            minimum=1, maximum=modules.config.default_max_image_number, step=1,
                            value=modules.config.default_image_number,
                            scale=3,
                            interactive=True
                        )
                        image_number_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_max_images = gr.Slider(
                            label='Max Image Number',
                            minimum=1, maximum=100, step=1,
                            value=modules.config.default_max_image_number,
                            scale=3,
                            interactive=True
                        )
                        max_images_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_output_format = gr.Radio(
                            label='Default Output Format',
                            choices=flags.OutputFormat.list(),
                            value=modules.config.default_output_format,
                            scale=3,
                            interactive=True
                        )
                        output_format_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                    with gr.Row():
                        config_aspect_ratio = gr.Radio(
                            label='Default Aspect Ratio',
                            choices=modules.config.available_aspect_ratios,
                            value=modules.config.default_aspect_ratio.replace('×', '*').split(' ')[0],
                            scale=3,
                            interactive=True
                        )
                        aspect_ratio_reset = gr.Button('↺', variant='secondary', scale=1, elem_classes=['reset-btn-mini'])
                
                # --- UI Defaults Section ---
                with gr.Accordion(label='🔧 UI Defaults', open=False, elem_classes=['config-section']):
                    with gr.Row():
                        config_advanced_cb = gr.Checkbox(
                            label='Show Advanced Panel by Default',
                            value=modules.config.default_advanced_checkbox,
                            interactive=True
                        )
                        advanced_cb_reset = gr.Button('↺ Reset', variant='secondary', elem_classes=['reset-btn'])
                    with gr.Row():
                        config_debug_mode = gr.Checkbox(
                            label='Developer Debug Mode by Default',
                            value=modules.config.default_developer_debug_mode_checkbox,
                            interactive=True
                        )
                        debug_mode_reset = gr.Button('↺ Reset', variant='secondary', elem_classes=['reset-btn'])
                    with gr.Row():
                        config_save_metadata = gr.Checkbox(
                            label='Save Metadata to Images by Default',
                            value=modules.config.default_save_metadata_to_images,
                            interactive=True
                        )
                        save_metadata_reset = gr.Button('↺ Reset', variant='secondary', elem_classes=['reset-btn'])
                    with gr.Row():
                        config_metadata_scheme = gr.Radio(
                            label='Default Metadata Scheme',
                            choices=flags.metadata_scheme,
                            value=modules.config.default_metadata_scheme,
                            visible=modules.config.default_save_metadata_to_images,
                            interactive=True
                        )
                    with gr.Row():
                        config_blackout_nsfw = gr.Checkbox(
                            label='Black Out NSFW by Default',
                            value=modules.config.default_black_out_nsfw,
                            interactive=True
                        )
                        blackout_nsfw_reset = gr.Button('↺ Reset', variant='secondary', elem_classes=['reset-btn'])
                
                # =========================================================================
                # Configuration Tab Event Handlers
                # =========================================================================
                
                def update_folder_display(folder_type):
                    """Get updated folder list for display."""
                    folders = modules.config.get_model_folders(folder_type)
                    return gr.update(value=[[p] for p in folders])
                
                def add_checkpoint_folder(folder_path):
                    """Add a checkpoint folder and return updated display and model dropdowns."""
                    if not folder_path or folder_path.strip() == '':
                        return gr.update(), gr.update(), gr.update()
                    
                    success, message, new_folders = modules.config.add_model_folder('path_checkpoints', folder_path.strip())
                    if success:
                        reload_result = modules.config.reload_model_files()
                        new_count = reload_result.get('model_count', 0)
                        print(f"✓ {message}. Found {new_count} new models.")
                        print(f"  Total models now: {len(modules.config.model_filenames)}")
                        print(f"  First few: {modules.config.model_filenames[:3]}")
                        return (
                            gr.update(value=[[p] for p in new_folders]),
                            gr.update(choices=modules.config.model_filenames),
                            gr.update(choices=['None'] + modules.config.model_filenames)
                        )
                    else:
                        print(f"✗ {message}")
                        return gr.update(), gr.update(), gr.update()
                
                def add_lora_folder(folder_path):
                    """Add a LoRA folder and return updated display."""
                    if not folder_path or folder_path.strip() == '':
                        # Return updates for folder display and all LoRA dropdowns (no change)
                        lora_updates = [gr.update() for _ in range(len(lora_ctrls))]
                        return (gr.update(), *lora_updates)
                    
                    success, message, new_folders = modules.config.add_model_folder('path_loras', folder_path.strip())
                    if success:
                        reload_result = modules.config.reload_model_files()
                        new_count = reload_result.get('lora_count', 0)
                        print(f"✓ {message}. Found {new_count} new LoRAs.")
                        # Return updates for folder display and all LoRA dropdowns
                        lora_updates = []
                        for i in range(modules.config.default_max_lora_number):
                            lora_updates.append(gr.update())  # enabled checkbox - no change
                            lora_updates.append(gr.update(choices=['None', modules.config.random_lora_name] + modules.config.lora_filenames))  # model dropdown
                            lora_updates.append(gr.update())  # weight slider - no change
                        return (
                            gr.update(value=[[p] for p in new_folders]),
                            *lora_updates
                        )
                    else:
                        print(f"✗ {message}")
                        lora_updates = [gr.update() for _ in range(len(lora_ctrls))]
                        return (gr.update(), *lora_updates)
                
                def reset_checkpoint_folders():
                    """Reset checkpoint folders to default."""
                    default = modules.config.get_default_config_value('path_checkpoints')
                    modules.config.config_dict['path_checkpoints'] = default
                    modules.config.reload_model_files()
                    return gr.update(value=[[p] for p in default])
                
                def reset_lora_folders():
                    """Reset LoRA folders to default."""
                    default = modules.config.get_default_config_value('path_loras')
                    modules.config.config_dict['path_loras'] = default
                    modules.config.reload_model_files()
                    return gr.update(value=[[p] for p in default])
                
                def reset_single_path(key, current_value):
                    """Reset a single path config to default."""
                    default = modules.config.get_default_config_value(key)
                    if default:
                        modules.config.config_dict[key] = default
                        return gr.update(value=default)
                    return gr.update()
                
                def reset_slider_value(key):
                    """Reset a slider value to default."""
                    default = modules.config.get_default_config_value(key)
                    if default is not None:
                        modules.config.config_dict[key] = default
                        return gr.update(value=default)
                    return gr.update()
                
                def reset_dropdown_value(key, choices=None):
                    """Reset a dropdown value to default."""
                    default = modules.config.get_default_config_value(key)
                    if default is not None:
                        modules.config.config_dict[key] = default
                        return gr.update(value=default)
                    return gr.update()
                
                def reset_checkbox_value(key):
                    """Reset a checkbox value to default."""
                    default = modules.config.get_default_config_value(key)
                    if default is not None:
                        modules.config.config_dict[key] = default
                        return gr.update(value=default)
                    return gr.update()
                
                def save_all_config():
                    """Save all configuration to file."""
                    if modules.config.save_config():
                        return '<p style="color: green; padding: 10px;">✓ Configuration saved successfully!</p>'
                    else:
                        return '<p style="color: red; padding: 10px;">✗ Failed to save configuration</p>'
                
                def restore_all_defaults():
                    """Restore all configuration to defaults."""
                    modules.config.restore_all_to_defaults()
                    modules.config.reload_model_files()
                    return '<p style="color: green; padding: 10px;">✓ All settings restored to defaults!</p>'
                
                # Wire up folder management events
                checkpoint_add_btn.click(
                    add_checkpoint_folder,
                    inputs=[checkpoint_folder_input],
                    outputs=[checkpoint_folders_display, base_model, refiner_model]
                )
                checkpoint_reset_btn.click(
                    reset_checkpoint_folders,
                    outputs=[checkpoint_folders_display]
                )
                
                lora_add_btn.click(
                    add_lora_folder,
                    inputs=[lora_folder_input],
                    outputs=[lora_folders_display] + lora_ctrls
                )
                lora_reset_btn.click(
                    reset_lora_folders,
                    outputs=[lora_folders_display]
                )
                
                # Wire up single path resets
                embeddings_reset.click(
                    lambda: reset_single_path('path_embeddings', modules.config.path_embeddings),
                    outputs=[embeddings_path]
                )
                vae_path_reset.click(
                    lambda: reset_single_path('path_vae', modules.config.path_vae),
                    outputs=[vae_path]
                )
                controlnet_reset.click(
                    lambda: reset_single_path('path_controlnet', modules.config.path_controlnet),
                    outputs=[controlnet_path]
                )
                upscale_reset.click(
                    lambda: reset_single_path('path_upscale_models', modules.config.path_upscale_models),
                    outputs=[upscale_path]
                )
                output_path_reset.click(
                    lambda: reset_single_path('path_outputs', modules.config.path_outputs),
                    outputs=[output_path]
                )
                temp_path_reset.click(
                    lambda: reset_single_path('temp_path', modules.config.temp_path),
                    outputs=[temp_path_config]
                )
                temp_cleanup_reset.click(
                    lambda: reset_checkbox_value('temp_path_cleanup_on_launch'),
                    outputs=[temp_cleanup]
                )
                
                # Wire up default model resets
                default_model_reset.click(
                    lambda: reset_dropdown_value('default_model'),
                    outputs=[config_default_model]
                )
                default_refiner_reset.click(
                    lambda: reset_dropdown_value('default_refiner'),
                    outputs=[config_default_refiner]
                )
                refiner_switch_reset.click(
                    lambda: reset_slider_value('default_refiner_switch'),
                    outputs=[config_refiner_switch]
                )
                default_vae_reset.click(
                    lambda: reset_dropdown_value('default_vae'),
                    outputs=[config_default_vae]
                )
                
                # Wire up generation settings resets
                default_steps_reset.click(
                    lambda: reset_slider_value('default_steps'),
                    outputs=[config_default_steps]
                )
                upscale_steps_reset.click(
                    lambda: reset_slider_value('default_upscale_steps'),
                    outputs=[config_upscale_steps]
                )
                cfg_scale_reset.click(
                    lambda: reset_slider_value('default_cfg_scale'),
                    outputs=[config_cfg_scale]
                )
                sharpness_reset.click(
                    lambda: reset_slider_value('default_sample_sharpness'),
                    outputs=[config_sharpness]
                )
                sampler_reset.click(
                    lambda: reset_dropdown_value('default_sampler'),
                    outputs=[config_sampler]
                )
                scheduler_reset.click(
                    lambda: reset_dropdown_value('default_scheduler'),
                    outputs=[config_scheduler]
                )
                clip_skip_reset.click(
                    lambda: reset_slider_value('default_clip_skip'),
                    outputs=[config_clip_skip]
                )
                adaptive_cfg_reset.click(
                    lambda: reset_slider_value('default_cfg_tsnr'),
                    outputs=[config_adaptive_cfg]
                )
                
                # Wire up styles reset
                default_styles_reset.click(
                    lambda: reset_dropdown_value('default_styles'),
                    outputs=[config_default_styles]
                )
                
                # Wire up image settings resets
                image_number_reset.click(
                    lambda: reset_slider_value('default_image_number'),
                    outputs=[config_image_number]
                )
                max_images_reset.click(
                    lambda: reset_slider_value('default_max_image_number'),
                    outputs=[config_max_images]
                )
                output_format_reset.click(
                    lambda: reset_dropdown_value('default_output_format'),
                    outputs=[config_output_format]
                )
                aspect_ratio_reset.click(
                    lambda: reset_dropdown_value('default_aspect_ratio'),
                    outputs=[config_aspect_ratio]
                )
                
                # Wire up UI defaults resets
                advanced_cb_reset.click(
                    lambda: reset_checkbox_value('default_advanced_checkbox'),
                    outputs=[config_advanced_cb]
                )
                debug_mode_reset.click(
                    lambda: reset_checkbox_value('default_developer_debug_mode_checkbox'),
                    outputs=[config_debug_mode]
                )
                save_metadata_reset.click(
                    lambda: reset_checkbox_value('default_save_metadata_to_images'),
                    outputs=[config_save_metadata]
                )
                blackout_nsfw_reset.click(
                    lambda: reset_checkbox_value('default_black_out_nsfw'),
                    outputs=[config_blackout_nsfw]
                )
                
                # Wire up save/restore all
                save_config_btn.click(save_all_config, outputs=[config_status])
                restore_all_btn.click(restore_all_defaults, outputs=[config_status])
                
                # Auto-save on config changes
                def auto_save_config(key, value):
                    """Update config value and auto-save."""
                    modules.config.update_config_value(key, value)
                    modules.config.save_config()
                    return ''
                
                # Wire auto-save for key settings
                config_default_model.change(lambda v: auto_save_config('default_model', v), inputs=[config_default_model])
                config_default_refiner.change(lambda v: auto_save_config('default_refiner', v), inputs=[config_default_refiner])
                config_refiner_switch.change(lambda v: auto_save_config('default_refiner_switch', v), inputs=[config_refiner_switch])
                config_default_vae.change(lambda v: auto_save_config('default_vae', v), inputs=[config_default_vae])
                config_default_steps.change(lambda v: auto_save_config('default_steps', v), inputs=[config_default_steps])
                config_upscale_steps.change(lambda v: auto_save_config('default_upscale_steps', v), inputs=[config_upscale_steps])
                config_cfg_scale.change(lambda v: auto_save_config('default_cfg_scale', v), inputs=[config_cfg_scale])
                config_sharpness.change(lambda v: auto_save_config('default_sample_sharpness', v), inputs=[config_sharpness])
                config_sampler.change(lambda v: auto_save_config('default_sampler', v), inputs=[config_sampler])
                config_scheduler.change(lambda v: auto_save_config('default_scheduler', v), inputs=[config_scheduler])
                config_clip_skip.change(lambda v: auto_save_config('default_clip_skip', v), inputs=[config_clip_skip])
                config_adaptive_cfg.change(lambda v: auto_save_config('default_cfg_tsnr', v), inputs=[config_adaptive_cfg])
                config_default_styles.change(lambda v: auto_save_config('default_styles', v), inputs=[config_default_styles])
                config_image_number.change(lambda v: auto_save_config('default_image_number', v), inputs=[config_image_number])
                config_max_images.change(lambda v: auto_save_config('default_max_image_number', v), inputs=[config_max_images])
                config_output_format.change(lambda v: auto_save_config('default_output_format', v), inputs=[config_output_format])
                config_aspect_ratio.change(lambda v: auto_save_config('default_aspect_ratio', v.replace('*', '×')), inputs=[config_aspect_ratio])
                config_advanced_cb.change(lambda v: auto_save_config('default_advanced_checkbox', v), inputs=[config_advanced_cb])
                config_debug_mode.change(lambda v: auto_save_config('default_developer_debug_mode_checkbox', v), inputs=[config_debug_mode])
                config_save_metadata.change(lambda v: auto_save_config('default_save_metadata_to_images', v), inputs=[config_save_metadata])
                config_metadata_scheme.change(lambda v: auto_save_config('default_metadata_scheme', v), inputs=[config_metadata_scheme])
                config_blackout_nsfw.change(lambda v: auto_save_config('default_black_out_nsfw', v), inputs=[config_blackout_nsfw])
                
                # Wire auto-save for path settings
                embeddings_path.change(lambda v: auto_save_config('path_embeddings', v), inputs=[embeddings_path])
                vae_path.change(lambda v: auto_save_config('path_vae', v), inputs=[vae_path])
                controlnet_path.change(lambda v: auto_save_config('path_controlnet', v), inputs=[controlnet_path])
                upscale_path.change(lambda v: auto_save_config('path_upscale_models', v), inputs=[upscale_path])
                output_path.change(lambda v: auto_save_config('path_outputs', v), inputs=[output_path])
                temp_path_config.change(lambda v: auto_save_config('temp_path', v), inputs=[temp_path_config])
                temp_cleanup.change(lambda v: auto_save_config('temp_path_cleanup_on_launch', v), inputs=[temp_cleanup])

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             steps_slider, upscale_steps_slider, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


            def inpaint_engine_state_change(inpaint_engine_version, *args):
                if inpaint_engine_version == 'empty':
                    inpaint_engine_version = modules.config.default_inpaint_engine_version

                result = []
                for inpaint_mode in args:
                    if inpaint_mode != modules.flags.inpaint_option_detail:
                        result.append(gr.update(value=inpaint_engine_version))
                    else:
                        result.append(gr.update())

                return result

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)



        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            steps_slider, upscale_steps_slider, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
