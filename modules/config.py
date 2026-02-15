import os
import json
import math
import numbers

import args_manager
import tempfile
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.extra_utils import makedirs_with_log, get_files_from_folder, try_eval_env_var
from modules.flags import OutputFormat, MetadataScheme


def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        print(f"Environment: {key} = {env}")
        return env
    else:
        return os.path.abspath(default_value)

wildcards_max_bfs_depth = 64
config_path = get_config_path('config_path', "./config.txt")
config_example_path = get_config_path('config_example_path', "config_modification_tutorial.txt")
config_dict = {}
always_save_keys = []
visited_keys = []

try:
    with open(os.path.abspath(f'./presets/default.json'), "r", encoding="utf-8") as json_file:
        config_dict.update(json.load(json_file))
except Exception as e:
    print(f'Load default preset failed.')
    print(e)

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
            always_save_keys = list(config_dict.keys())
except Exception as e:
    print(f'Failed to load config file "{config_path}" . The reason is: {str(e)}')
    print('Please make sure that:')
    print(f'1. The file "{config_path}" is a valid text file, and you have access to read it.')
    print('2. Use "\\\\" instead of "\\" when describing paths.')
    print('3. There is no "," before the last "}".')
    print('4. All key/value formats are correct.')


def try_load_deprecated_user_path_config():
    global config_dict

    if not os.path.exists('user_path_config.txt'):
        return

    try:
        deprecated_config_dict = json.load(open('user_path_config.txt', "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                config_dict[new_key] = deprecated_config_dict[old_key]
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'path_checkpoints')
        replace_config('lorafile_path', 'path_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')
        replace_config('fooocus_expansion_path', 'path_fooocus_expansion')
        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_version6Rundiffusion.safetensors':
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            config_dict.update(deprecated_config_dict)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


try_load_deprecated_user_path_config()

def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index(".json")] for f in os.listdir(preset_folder) if f.endswith('.json')]

def update_presets():
    global available_presets
    available_presets = get_presets()

def try_get_preset_content(preset):
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    json_content = json.load(json_file)
                    print(f'Loaded preset: {preset_path}')
                    return json_content
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    return {}

available_presets = get_presets()
preset = args_manager.args.preset
config_dict.update(try_get_preset_content(preset))

def get_path_output() -> str:
    """
    Checking output path argument and overriding default path.
    """
    global config_dict
    path_output = get_dir_or_set_default('path_outputs', '../outputs/', make_directory=True)
    if args_manager.args.output_path:
        print(f'Overriding config value path_outputs with {args_manager.args.output_path}')
        config_dict['path_outputs'] = path_output = args_manager.args.output_path
    return path_output


def get_dir_or_set_default(key, default_value, as_array=False, make_directory=False):
    global config_dict, visited_keys, always_save_keys

    if key not in visited_keys:
        visited_keys.append(key)

    if key not in always_save_keys:
        always_save_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v
    else:
        v = config_dict.get(key, None)

    if isinstance(v, str):
        if make_directory:
            makedirs_with_log(v)
        if os.path.exists(v) and os.path.isdir(v):
            return v if not as_array else [v]
    elif isinstance(v, list):
        if make_directory:
            for d in v:
                makedirs_with_log(d)
        if all([os.path.exists(d) and os.path.isdir(d) for d in v]):
            return v

    if v is not None:
        print(f'Failed to load config key: {json.dumps({key:v})} is invalid or does not exist; will use {json.dumps({key:default_value})} instead.')
    if isinstance(default_value, list):
        dp = []
        for path in default_value:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            dp.append(abs_path)
            os.makedirs(abs_path, exist_ok=True)
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        if as_array:
            dp = [dp]
    config_dict[key] = dp
    return dp


paths_checkpoints = get_dir_or_set_default('path_checkpoints', ['../models/checkpoints/'], True)
paths_loras = get_dir_or_set_default('path_loras', ['../models/loras/'], True)
path_embeddings = get_dir_or_set_default('path_embeddings', '../models/embeddings/')
path_vae_approx = get_dir_or_set_default('path_vae_approx', '../models/vae_approx/')
path_vae = get_dir_or_set_default('path_vae', '../models/vae/')
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/')
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/')
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/')
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/clip_vision/')
path_fooocus_expansion = get_dir_or_set_default('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion')
path_wildcards = get_dir_or_set_default('path_wildcards', '../wildcards/')
path_safety_checker = get_dir_or_set_default('path_safety_checker', '../models/safety_checker/')
path_sam = get_dir_or_set_default('path_sam', '../models/sam/')
path_outputs = get_path_output()


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False, expected_type=None):
    global config_dict, visited_keys

    if key not in visited_keys:
        visited_keys.append(key)
    
    v = os.getenv(key)
    if v is not None:
        v = try_eval_env_var(v, expected_type)
        print(f"Environment: {key} = {v}")
        config_dict[key] = v

    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        if v is not None:
            print(f'Failed to load config key: {json.dumps({key:v})} is invalid; will use {json.dumps({key:default_value})} instead.')
        config_dict[key] = default_value
        return default_value


def init_temp_path(path: str | None, default_path: str) -> str:
    if args_manager.args.temp_path:
        path = args_manager.args.temp_path

    if path != '' and path != default_path:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            print(f'Using temp path {path}')
            return path
        except Exception as e:
            print(f'Could not create temp path {path}. Reason: {e}')
            print(f'Using default temp path {default_path} instead.')

    os.makedirs(default_path, exist_ok=True)
    return default_path


default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')
temp_path = init_temp_path(get_config_item_or_set_default(
    key='temp_path',
    default_value=default_temp_path,
    validator=lambda x: isinstance(x, str),
    expected_type=str
), default_temp_path)
temp_path_cleanup_on_launch = get_config_item_or_set_default(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_base_model_name = default_model = get_config_item_or_set_default(
    key='default_model',
    default_value='model.safetensors',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
previous_default_models = get_config_item_or_set_default(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x),
    expected_type=list
)
default_refiner_model_name = default_refiner = get_config_item_or_set_default(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_refiner_switch = get_config_item_or_set_default(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1,
    expected_type=numbers.Number
)
default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=-2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras_max_weight = get_config_item_or_set_default(
    key='default_loras_max_weight',
    default_value=2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras = get_config_item_or_set_default(
    key='default_loras',
    default_value=[
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x),
    expected_type=list
)
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_item_or_set_default(
    key='default_max_lora_number',
    default_value=len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else 5,
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list,
    expected_type=str
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list,
    expected_type=str
)
default_vae = get_config_item_or_set_default(
    key='default_vae',
    default_value=modules.flags.default_vae,
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=[
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp"
    ],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x),
    expected_type=list
)
default_prompt_negative = get_config_item_or_set_default(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_prompt = get_config_item_or_set_default(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_steps = get_config_item_or_set_default(
    key='default_steps',
    default_value=25,
    validator=lambda x: isinstance(x, int) and 1 <= x <= 200,
    expected_type=int
)
default_upscale_steps = get_config_item_or_set_default(
    key='default_upscale_steps',
    default_value=20,
    validator=lambda x: isinstance(x, int) and 1 <= x <= 200,
    expected_type=int
)
default_image_prompt_checkbox = get_config_item_or_set_default(
    key='default_image_prompt_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_enhance_checkbox = get_config_item_or_set_default(
    key='default_enhance_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_developer_debug_mode_checkbox = get_config_item_or_set_default(
    key='default_developer_debug_mode_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_image_prompt_advanced_checkbox = get_config_item_or_set_default(
    key='default_image_prompt_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_max_image_number = get_config_item_or_set_default(
    key='default_max_image_number',
    default_value=32,
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_output_format = get_config_item_or_set_default(
    key='default_output_format',
    default_value='png',
    validator=lambda x: x in OutputFormat.list(),
    expected_type=str
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number,
    expected_type=int
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
vae_downloads = get_config_item_or_set_default(
    key='vae_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=modules.flags.sdxl_aspect_ratios,
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1,
    expected_type=list
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios,
    expected_type=str
)
default_inpaint_engine_version = get_config_item_or_set_default(
    key='default_inpaint_engine_version',
    default_value='v2.6',
    validator=lambda x: x in modules.flags.inpaint_engine_versions,
    expected_type=str
)
default_selected_image_input_tab_id = get_config_item_or_set_default(
    key='default_selected_image_input_tab_id',
    default_value=modules.flags.default_input_image_tab,
    validator=lambda x: x in modules.flags.input_image_tab_ids,
    expected_type=str
)
default_uov_method = get_config_item_or_set_default(
    key='default_uov_method',
    default_value=modules.flags.disabled,
    validator=lambda x: x in modules.flags.uov_list,
    expected_type=str
)
default_controlnet_image_count = get_config_item_or_set_default(
    key='default_controlnet_image_count',
    default_value=4,
    validator=lambda x: isinstance(x, int) and x > 0,
    expected_type=int
)
default_ip_images = {}
default_ip_stop_ats = {}
default_ip_weights = {}
default_ip_types = {}

for image_count in range(default_controlnet_image_count):
    image_count += 1
    default_ip_images[image_count] = get_config_item_or_set_default(
        key=f'default_ip_image_{image_count}',
        default_value='None',
        validator=lambda x: x == 'None' or isinstance(x, str) and os.path.exists(x),
        expected_type=str
    )

    if default_ip_images[image_count] == 'None':
        default_ip_images[image_count] = None

    default_ip_types[image_count] = get_config_item_or_set_default(
        key=f'default_ip_type_{image_count}',
        default_value=modules.flags.default_ip,
        validator=lambda x: x in modules.flags.ip_list,
        expected_type=str
    )

    default_end, default_weight = modules.flags.default_parameters[default_ip_types[image_count]]

    default_ip_stop_ats[image_count] = get_config_item_or_set_default(
        key=f'default_ip_stop_at_{image_count}',
        default_value=default_end,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 1,
        expected_type=float
    )
    default_ip_weights[image_count] = get_config_item_or_set_default(
        key=f'default_ip_weight_{image_count}',
        default_value=default_weight,
        validator=lambda x: isinstance(x, float) and 0 <= x <= 2,
        expected_type=float
    )

default_inpaint_advanced_masking_checkbox = get_config_item_or_set_default(
    key='default_inpaint_advanced_masking_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_inpaint_method = get_config_item_or_set_default(
    key='default_inpaint_method',
    default_value=modules.flags.inpaint_option_default,
    validator=lambda x: x in modules.flags.inpaint_options,
    expected_type=str
)
default_cfg_tsnr = get_config_item_or_set_default(
    key='default_cfg_tsnr',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_clip_skip = get_config_item_or_set_default(
    key='default_clip_skip',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= modules.flags.clip_skip_max,
    expected_type=int
)
default_overwrite_step = get_config_item_or_set_default(
    key='default_overwrite_step',
    default_value=-1,
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_switch = get_config_item_or_set_default(
    key='default_overwrite_switch',
    default_value=-1,
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_upscale = get_config_item_or_set_default(
    key='default_overwrite_upscale',
    default_value=-1,
    validator=lambda x: isinstance(x, numbers.Number)
)
example_inpaint_prompts = get_config_item_or_set_default(
    key='example_inpaint_prompts',
    default_value=[
        'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
    ],
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
    expected_type=list
)
example_enhance_detection_prompts = get_config_item_or_set_default(
    key='example_enhance_detection_prompts',
    default_value=[
        'face', 'eye', 'mouth', 'hair', 'hand', 'body'
    ],
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
    expected_type=list
)
default_enhance_tabs = get_config_item_or_set_default(
    key='default_enhance_tabs',
    default_value=3,
    validator=lambda x: isinstance(x, int) and 1 <= x <= 5,
    expected_type=int
)
default_enhance_uov_method = get_config_item_or_set_default(
    key='default_enhance_uov_method',
    default_value=modules.flags.disabled,
    validator=lambda x: x in modules.flags.uov_list,
    expected_type=int
)
default_enhance_uov_processing_order = get_config_item_or_set_default(
    key='default_enhance_uov_processing_order',
    default_value=modules.flags.enhancement_uov_before,
    validator=lambda x: x in modules.flags.enhancement_uov_processing_order,
    expected_type=int
)
default_enhance_uov_prompt_type = get_config_item_or_set_default(
    key='default_enhance_uov_prompt_type',
    default_value=modules.flags.enhancement_uov_prompt_type_original,
    validator=lambda x: x in modules.flags.enhancement_uov_prompt_types,
    expected_type=int
)
default_sam_max_detections = get_config_item_or_set_default(
    key='default_sam_max_detections',
    default_value=0,
    validator=lambda x: isinstance(x, int) and 0 <= x <= 10,
    expected_type=int
)
default_black_out_nsfw = get_config_item_or_set_default(
    key='default_black_out_nsfw',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_save_only_final_enhanced_image = get_config_item_or_set_default(
    key='default_save_only_final_enhanced_image',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_save_metadata_to_images = get_config_item_or_set_default(
    key='default_save_metadata_to_images',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_metadata_scheme = get_config_item_or_set_default(
    key='default_metadata_scheme',
    default_value=MetadataScheme.FOOOCUS.value,
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x],
    expected_type=str
)
metadata_created_by = get_config_item_or_set_default(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)

example_inpaint_prompts = [[x] for x in example_inpaint_prompts]
example_enhance_detection_prompts = [[x] for x in example_enhance_detection_prompts]

default_invert_mask_checkbox = get_config_item_or_set_default(
    key='default_invert_mask_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)

default_inpaint_mask_model = get_config_item_or_set_default(
    key='default_inpaint_mask_model',
    default_value='isnet-general-use',
    validator=lambda x: x in modules.flags.inpaint_mask_models,
    expected_type=str
)

default_enhance_inpaint_mask_model = get_config_item_or_set_default(
    key='default_enhance_inpaint_mask_model',
    default_value='sam',
    validator=lambda x: x in modules.flags.inpaint_mask_models,
    expected_type=str
)

default_inpaint_mask_cloth_category = get_config_item_or_set_default(
    key='default_inpaint_mask_cloth_category',
    default_value='full',
    validator=lambda x: x in modules.flags.inpaint_mask_cloth_category,
    expected_type=str
)

default_inpaint_mask_sam_model = get_config_item_or_set_default(
    key='default_inpaint_mask_sam_model',
    default_value='vit_b',
    validator=lambda x: x in modules.flags.inpaint_mask_sam_model,
    expected_type=str
)

default_describe_apply_prompts_checkbox = get_config_item_or_set_default(
    key='default_describe_apply_prompts_checkbox',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_describe_content_type = get_config_item_or_set_default(
    key='default_describe_content_type',
    default_value=[modules.flags.describe_type_photo],
    validator=lambda x: all(k in modules.flags.describe_types for k in x),
    expected_type=list
)

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]

# mapping config to meta parameter
possible_preset_keys = {
    "default_model": "base_model",
    "default_refiner": "refiner_model",
    "default_refiner_switch": "refiner_switch",
    "previous_default_models": "previous_default_models",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_cfg_tsnr": "adaptive_cfg",
    "default_clip_skip": "clip_skip",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "overwrite_step",
    "default_overwrite_switch": "overwrite_switch",
    "default_steps": "steps",
    "default_upscale_steps": "upscale_steps",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads",
    "vae_downloads": "vae_downloads",
    "default_vae": "vae",
    # "default_inpaint_method": "inpaint_method", # disabled so inpaint mode doesn't refresh after every preset change
    "default_inpaint_engine_version": "inpaint_engine_version",
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = 'presets/' + args_manager.args.preset + '.json'
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
    print(f'Preset saved to {save_path}. Exiting ...')
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {a // g}:{b // g}</span>'


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios_labels = [add_ratio(x) for x in available_aspect_ratios]


# Only write config in the first launch.
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in always_save_keys}, json_file, indent=4)


# Always write tutorials.
with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

model_filenames = []
lora_filenames = []
vae_filenames = []
wildcard_filenames = []


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        try:
            files += get_files_from_folder(folder, extensions, name_filter)
        except (ValueError, FileNotFoundError) as e:
            print(f"  [Config] Skipping invalid folder: {folder} ({e})")
            continue

    return files


def update_files():
    global model_filenames, lora_filenames, vae_filenames, wildcard_filenames, available_presets
    model_filenames = get_model_filenames(paths_checkpoints)
    lora_filenames = get_model_filenames(paths_loras)
    vae_filenames = get_model_filenames(path_vae)
    wildcard_filenames = get_files_from_folder(path_wildcards, ['.txt'])
    available_presets = get_presets()
    return


# Random LoRA functionality
random_lora_name = 'Random LoRA'


def get_random_lora(rng):
    """Get a random LoRA from available LoRA files, similar to get_random_style"""
    if not lora_filenames:
        return 'None'
    return rng.choice(lora_filenames)


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.EXTREME_SPEED.value
    )
    return modules.flags.PerformanceLoRA.EXTREME_SPEED.value


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.LIGHTNING.value
    )
    return modules.flags.PerformanceLoRA.LIGHTNING.value


def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.HYPER_SD.value
    )
    return modules.flags.PerformanceLoRA.HYPER_SD.value


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')

def downloading_safety_checker_model():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=path_safety_checker,
        file_name='stable-diffusion-safety-checker.bin'
    )
    return os.path.join(path_safety_checker, 'stable-diffusion-safety-checker.bin')


def download_sam_model(sam_model: str) -> str:
    match sam_model:
        case 'vit_b':
            return downloading_sam_vit_b()
        case 'vit_l':
            return downloading_sam_vit_l()
        case 'vit_h':
            return downloading_sam_vit_h()
        case _:
            raise ValueError(f"sam model {sam_model} does not exist.")


def downloading_sam_vit_b():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        model_dir=path_sam,
        file_name='sam_vit_b_01ec64.pth'
    )
    return os.path.join(path_sam, 'sam_vit_b_01ec64.pth')


def downloading_sam_vit_l():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        model_dir=path_sam,
        file_name='sam_vit_l_0b3195.pth'
    )
    return os.path.join(path_sam, 'sam_vit_l_0b3195.pth')


def downloading_sam_vit_h():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        model_dir=path_sam,
        file_name='sam_vit_h_4b8939.pth'
    )
    return os.path.join(path_sam, 'sam_vit_h_4b8939.pth')


# =============================================================================
# Configuration Management for UI
# =============================================================================

# Default configuration values for restore functionality
# These are the built-in defaults that settings can be restored to
DEFAULT_CONFIG = {
    # Model paths
    'path_checkpoints': ['../models/checkpoints/'],
    'path_loras': ['../models/loras/'],
    'path_embeddings': '../models/embeddings/',
    'path_vae': '../models/vae/',
    'path_vae_approx': '../models/vae_approx/',
    'path_upscale_models': '../models/upscale_models/',
    'path_inpaint': '../models/inpaint/',
    'path_controlnet': '../models/controlnet/',
    'path_clip_vision': '../models/clip_vision/',
    'path_fooocus_expansion': '../models/prompt_expansion/fooocus_expansion',
    'path_wildcards': '../wildcards/',
    'path_safety_checker': '../models/safety_checker/',
    'path_sam': '../models/sam/',
    'path_outputs': '../outputs/',
    
    # Default models
    'default_model': 'model.safetensors',
    'default_refiner': 'None',
    'default_refiner_switch': 0.8,
    'default_vae': modules.flags.default_vae,
    
    # Default LoRAs
    'default_loras': [
        [True, 'None', 1.0],
        [True, 'None', 1.0],
        [True, 'None', 1.0],
        [True, 'None', 1.0],
        [True, 'None', 1.0]
    ],
    'default_loras_min_weight': -2,
    'default_loras_max_weight': 2,
    'default_max_lora_number': 5,
    
    # Generation settings
    'default_steps': 25,
    'default_upscale_steps': 20,
    'default_cfg_scale': 7.0,
    'default_sample_sharpness': 2.0,
    'default_sampler': 'dpmpp_2m_sde_gpu',
    'default_scheduler': 'karras',
    'default_cfg_tsnr': 7.0,
    'default_clip_skip': 2,
    
    # Styles and prompts
    'default_styles': ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp'],
    'default_prompt': '',
    'default_prompt_negative': '',
    
    # Image settings
    'default_image_number': 2,
    'default_max_image_number': 32,
    'default_output_format': 'png',
    'available_aspect_ratios': modules.flags.sdxl_aspect_ratios,
    'default_aspect_ratio': '1152*896',
    
    # UI defaults
    'default_advanced_checkbox': True,
    'default_developer_debug_mode_checkbox': False,
    'default_image_prompt_checkbox': False,
    'default_enhance_checkbox': False,
    'default_image_prompt_advanced_checkbox': False,
    'default_inpaint_advanced_masking_checkbox': False,
    'default_inpaint_method': modules.flags.inpaint_option_default,
    'default_inpaint_engine_version': 'v2.6',
    
    # Additional settings
    'default_save_metadata_to_images': False,
    'default_metadata_scheme': MetadataScheme.FOOOCUS.value,
    'default_black_out_nsfw': False,
    'default_save_only_final_enhanced_image': False,
    'default_overwrite_step': -1,
    'default_overwrite_switch': -1,
    'default_overwrite_upscale': -1,
    'temp_path_cleanup_on_launch': True,
    
    # Downloads
    'checkpoint_downloads': {},
    'lora_downloads': {},
    'embeddings_downloads': {},
    'vae_downloads': {},
    'previous_default_models': [],
}


def get_default_config_value(key):
    """Get the default value for a config key.
    
    Args:
        key: The configuration key to look up
        
    Returns:
        The default value for the key, or None if key not found
    """
    return DEFAULT_CONFIG.get(key)


def save_config(specific_keys=None):
    """Save the current configuration to the config file.
    
    Args:
        specific_keys: Optional list of keys to save. If None, saves all keys in always_save_keys.
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    global config_dict, always_save_keys
    
    try:
        keys_to_save = specific_keys if specific_keys else always_save_keys
        # Filter to only include keys that exist in config_dict
        config_to_save = {k: config_dict[k] for k in keys_to_save if k in config_dict}
        
        print(f"[Config] Saving {len(config_to_save)} keys to {config_path}")
        print(f"[Config] Keys: {list(config_to_save.keys())}")
        
        with open(config_path, "w", encoding="utf-8") as json_file:
            json.dump(config_to_save, json_file, indent=4)
        
        print(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Failed to save configuration: {e}")
        return False


def update_config_value(key, value):
    """Update a single configuration value and optionally save.
    
    Args:
        key: The configuration key to update
        value: The new value
        
    Returns:
        bool: True if update was successful
    """
    global config_dict, always_save_keys
    
    print(f"[Config] Updating {key} = {value}")
    config_dict[key] = value
    
    if key not in always_save_keys:
        always_save_keys.append(key)
    
    return True


def restore_config_to_default(key):
    """Restore a specific configuration key to its default value.
    
    Args:
        key: The configuration key to restore
        
    Returns:
        The default value that was restored, or None if key not found
    """
    global config_dict, always_save_keys
    
    default_value = get_default_config_value(key)
    if default_value is not None:
        config_dict[key] = default_value
        
        if key not in always_save_keys:
            always_save_keys.append(key)
            
        return default_value
    
    return None


def restore_all_to_defaults():
    """Restore all configuration values to their defaults.
    
    Returns:
        bool: True if successful
    """
    global config_dict, always_save_keys
    
    for key, value in DEFAULT_CONFIG.items():
        config_dict[key] = value
        if key not in always_save_keys:
            always_save_keys.append(key)
    
    save_config()
    return True


def add_model_folder(folder_type, folder_path):
    """Add a model folder to a multi-folder path config.
    
    Args:
        folder_type: The config key for the folder type (e.g., 'path_checkpoints')
        folder_path: The path to add
        
    Returns:
        tuple: (success: bool, message: str, new_folders: list)
    """
    global config_dict
    
    # Validate folder path
    if not os.path.exists(folder_path):
        return False, f"Folder does not exist: {folder_path}", None
    
    if not os.path.isdir(folder_path):
        return False, f"Path is not a directory: {folder_path}", None
    
    # Get current folders
    current = config_dict.get(folder_type, [])
    
    # Handle single-path configs by converting to list
    if isinstance(current, str):
        current = [current]
    elif not isinstance(current, list):
        current = [current]
    
    # Check if already exists
    abs_path = os.path.abspath(folder_path)
    current_abs = [os.path.abspath(p) for p in current]
    
    if abs_path in current_abs:
        return False, f"Folder already in list: {folder_path}", current
    
    # Add new folder
    new_folders = current + [folder_path]
    config_dict[folder_type] = new_folders
    
    # Save config to make it persistent
    save_config()
    
    return True, f"Added folder: {folder_path}", new_folders


def remove_model_folder(folder_type, folder_path):
    """Remove a model folder from a multi-folder path config.
    
    Args:
        folder_type: The config key for the folder type
        folder_path: The path to remove
        
    Returns:
        tuple: (success: bool, message: str, new_folders: list)
    """
    global config_dict
    
    current = config_dict.get(folder_type, [])
    
    if isinstance(current, str):
        current = [current]
    elif not isinstance(current, list):
        current = [current]
    
    # Find and remove the folder
    abs_path = os.path.abspath(folder_path)
    new_folders = [p for p in current if os.path.abspath(p) != abs_path]
    
    if len(new_folders) == len(current):
        return False, f"Folder not found: {folder_path}", current
    
    # Ensure at least one folder remains
    if len(new_folders) == 0:
        default_folders = get_default_config_value(folder_type)
        if default_folders:
            new_folders = [default_folders] if isinstance(default_folders, str) else default_folders
        else:
            return False, "Cannot remove the last folder", current
    
    config_dict[folder_type] = new_folders
    
    # Save config to make it persistent
    save_config()
    
    return True, f"Removed folder: {folder_path}", new_folders


def get_model_folders(folder_type):
    """Get the list of folders for a model type.
    
    Args:
        folder_type: The config key for the folder type
        
    Returns:
        list: List of folder paths
    """
    global config_dict
    
    current = config_dict.get(folder_type, [])
    
    if isinstance(current, str):
        return [current]
    elif isinstance(current, list):
        return current
    
    return []


def reload_model_files():
    """Reload model files from all configured folders.
    
    This function scans all model folders and updates the file lists.
    
    Returns:
        dict: Dictionary with counts of new files found per type
    """
    global model_filenames, lora_filenames, vae_filenames, paths_checkpoints, paths_loras
    
    # Update global path variables from config_dict
    # This ensures newly added folders are included
    paths_checkpoints = get_model_folders('path_checkpoints')
    paths_loras = get_model_folders('path_loras')
    
    old_models = set(model_filenames)
    old_loras = set(lora_filenames)
    old_vaes = set(vae_filenames)
    
    update_files()
    
    new_models = set(model_filenames) - old_models
    new_loras = set(lora_filenames) - old_loras
    new_vaes = set(vae_filenames) - old_vaes
    
    return {
        'models': list(new_models),
        'loras': list(new_loras),
        'vaes': list(new_vaes),
        'model_count': len(new_models),
        'lora_count': len(new_loras),
        'vae_count': len(new_vaes),
        'total_new': len(new_models) + len(new_loras) + len(new_vaes)
    }
