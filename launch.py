import os
import ssl
import sys
import importlib.metadata
import packaging.version

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = True


def torch_stack_is_compatible(min_torch_version: str = "2.2.0") -> bool:
    try:
        torch_version_raw = importlib.metadata.version("torch")
        torch_version = torch_version_raw.split("+", 1)[0]
        return packaging.version.parse(torch_version) >= packaging.version.parse(min_torch_version)
    except Exception:
        return False


def _installed_dist_version(package: str):
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def _recommended_xformers_for_torch() -> str | None:
    torch_version_raw = _installed_dist_version("torch")
    if not torch_version_raw:
        return None
    torch_version = torch_version_raw.split("+", 1)[0]
    mapping = {
        "2.5.0": "xformers==0.0.28.post2",
        "2.5.1": "xformers==0.0.29.post1",
        "2.6.0": "xformers==0.0.29.post2",
        "2.9.0": "xformers==0.0.33.post1",
        "2.9.1": "xformers==0.0.33.post2",
        "2.10.0": "xformers==0.0.34",
    }
    return mapping.get(torch_version, None)


def _version_from_pin(spec: str) -> str | None:
    if "==" not in spec:
        return None
    return spec.split("==", 1)[1].strip() or None


def _xformers_runtime_healthy() -> bool:
    try:
        import xformers  # noqa: F401
        # Touch CUDA extension path to catch ABI mismatches early.
        import xformers.ops  # noqa: F401
        return True
    except Exception as e:
        print(f"xformers runtime check failed: {e}")
        return False


def _cleanup_incompatible_xformers() -> None:
    try:
        run(
            f'"{python}" -m pip uninstall -y xformers',
            desc="Removing incompatible xformers",
            errdesc="Couldn't uninstall incompatible xformers",
            live=True,
        )
    except Exception as e:
        print(e)

    # If metadata is corrupted, pip uninstall may leave files behind.
    # Best effort: remove lingering xformers package files directly.
    try:
        import glob
        import shutil
        import site

        roots = []
        try:
            roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                roots.append(user_site)
        except Exception:
            pass

        removed_any = False
        for root in roots:
            for path in glob.glob(os.path.join(root, "xformers*")):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        removed_any = True
                    elif os.path.isfile(path):
                        os.remove(path)
                        removed_any = True
                except Exception as sub_e:
                    print(f"Failed to remove leftover xformers path {path}: {sub_e}")
        if removed_any:
            print("Removed leftover xformers files from site-packages.")
    except Exception as e:
        print(f"Best-effort xformers file cleanup skipped: {e}")


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.9.1 torchvision==0.24.1 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    need_torch_stack_install = (
        REINSTALL_ALL
        or not is_installed("torch")
        or not is_installed("torchvision")
        or not torch_stack_is_compatible("2.9.1")
    )
    if need_torch_stack_install:
        print("Installing/upgrading torch stack (requires torch>=2.9.1 for this safe branch profile).")
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        xformers_package = os.environ.get('XFORMERS_PACKAGE', "").strip()
        if not xformers_package:
            xformers_package = _recommended_xformers_for_torch() or ""

        if not xformers_package:
            print("No pinned xformers package for current torch version. Skipping xformers auto-install.")
        else:
            expected_xformers_version = _version_from_pin(xformers_package)
            installed_xformers_version = _installed_dist_version("xformers")
            need_xformers_install = (
                REINSTALL_ALL
                or installed_xformers_version is None
                or (
                    expected_xformers_version is not None
                    and installed_xformers_version.split("+", 1)[0] != expected_xformers_version
                )
            )
            if installed_xformers_version and need_xformers_install:
                print(
                    f"xformers {installed_xformers_version} is incompatible with current torch stack; "
                    f"reinstalling {xformers_package}."
                )
            if need_xformers_install:
                print(f"Installing pinned xformers package: {xformers_package} (no deps).")
                if platform.system() == "Windows":
                    if platform.python_version().startswith("3.10"):
                        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                    else:
                        print("Installation of xformers is not supported in this version of Python.")
                        print(
                            "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                        if not is_installed("xformers"):
                            exit(0)
                elif platform.system() == "Linux":
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

            if _installed_dist_version("xformers") is not None and not _xformers_runtime_healthy():
                print("xformers binary is incompatible with current torch/CUDA stack. Uninstalling xformers and continuing with PyTorch attention.")
                _cleanup_incompatible_xformers()

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

from webui import *
