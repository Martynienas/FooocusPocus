import json
import os
from typing import Optional


_PIPELINE_CACHE = {}


def is_zimage_checkpoint_name(name: str) -> bool:
    lowered = (name or "").lower()
    return "z-image" in lowered or "zimage" in lowered or "tongyi" in lowered


def detect_zimage_flavor(name: str) -> str:
    lowered = (name or "").lower()
    if "turbo" in lowered:
        return "turbo"
    return "standard"


def _repo_for_flavor(flavor: str) -> str:
    if flavor == "turbo":
        return "Tongyi-MAI/Z-Image-Turbo"
    return "Tongyi-MAI/Z-Image"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_zimage_model_directory(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    model_index_path = os.path.join(path, "model_index.json")
    if os.path.isfile(model_index_path):
        try:
            model_index = _read_json(model_index_path)
            class_name = str(model_index.get("_class_name", ""))
            if "zimage" in class_name.lower():
                return True
            model_entries = str(model_index.get("transformer", ""))
            if "zimage" in model_entries.lower():
                return True
        except Exception:
            pass

    transformer_config_path = os.path.join(path, "transformer", "config.json")
    if os.path.isfile(transformer_config_path):
        try:
            transformer_config = _read_json(transformer_config_path)
            class_name = str(transformer_config.get("_class_name", ""))
            dim = int(transformer_config.get("dim", -1))
            return "zimage" in class_name.lower() or dim == 3840
        except Exception:
            return False

    return False


def list_zimage_model_entries(checkpoint_folders: list[str]) -> list[str]:
    entries = []
    for folder in checkpoint_folders:
        if not os.path.isdir(folder):
            continue
        for root, dirs, _ in os.walk(folder, topdown=True):
            relative_root = os.path.relpath(root, folder)
            relative_root = "" if relative_root == "." else relative_root
            if is_zimage_model_directory(root):
                if relative_root:
                    entries.append(relative_root)
                dirs[:] = []
                continue
    return sorted(set(entries), key=str.casefold)


def resolve_zimage_model_path(name: str, checkpoint_folders: list[str]) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None

    if os.path.isabs(name) and is_zimage_model_directory(name):
        return name

    for folder in checkpoint_folders:
        candidate = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if is_zimage_model_directory(candidate):
            return candidate

    return None


def _resolve_named_path(name: str, checkpoint_folders: list[str]) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None

    if os.path.isabs(name) and os.path.exists(name):
        return os.path.abspath(os.path.realpath(name))

    for folder in checkpoint_folders:
        candidate = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.exists(candidate):
            return candidate

    # Fallback: match by basename anywhere inside configured checkpoint folders.
    target = os.path.basename(name).casefold()
    for folder in checkpoint_folders:
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder, topdown=True):
            for file_name in files:
                if file_name.casefold() == target:
                    return os.path.abspath(os.path.realpath(os.path.join(root, file_name)))
        for root, dirs, _ in os.walk(folder, topdown=True):
            for dir_name in dirs:
                if dir_name.casefold() == target:
                    return os.path.abspath(os.path.realpath(os.path.join(root, dir_name)))

    return None


def _is_likely_zimage_safetensors(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if not path.lower().endswith(".safetensors"):
        return False

    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            if any(k.startswith("text_encoders.qwen3_4b.") for k in keys):
                return True

            if "cap_embedder.1.weight" in keys:
                cap_shape = tuple(f.get_tensor("cap_embedder.1.weight").shape)
                if len(cap_shape) >= 1 and cap_shape[0] == 3840:
                    return True

            has_lumina_backbone = any(k.startswith("layers.0.attention.") for k in keys)
            has_zimage_text = any(k.startswith("text_encoders.") and "qwen3" in k for k in keys)
            if has_lumina_backbone and has_zimage_text:
                return True
    except Exception:
        return False

    return False


def should_use_zimage_checkpoint(name: str, checkpoint_folders: list[str]) -> bool:
    if is_zimage_checkpoint_name(name):
        return True

    resolved = _resolve_named_path(name, checkpoint_folders)
    if resolved is None:
        return False

    if os.path.isdir(resolved):
        return is_zimage_model_directory(resolved)

    return _is_likely_zimage_safetensors(resolved)


def _find_local_repo_components(flavor: str, checkpoint_folders: list[str]) -> Optional[str]:
    repo = _repo_for_flavor(flavor).split("/")[-1]
    candidates = [repo, repo.lower(), repo.replace("-", "_").lower()]

    for folder in checkpoint_folders:
        for cand in candidates:
            path = os.path.abspath(os.path.realpath(os.path.join(folder, cand)))
            if is_zimage_model_directory(path):
                return path
        # Also support nested mirrors like Tongyi-MAI/Z-Image-Turbo.
        nested = os.path.abspath(os.path.realpath(os.path.join(folder, "Tongyi-MAI", repo)))
        if is_zimage_model_directory(nested):
            return nested

    return None


def _load_transformer_weights_from_single_file(single_file_path: str, pipeline) -> None:
    from safetensors.torch import load_file

    sd = load_file(single_file_path, device="cpu")

    # Common AIO prefix from Forge/Comfy exports.
    prefix = "model.diffusion_model."
    transformer_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            transformer_sd[k[len(prefix):]] = v

    # Fallback for already-converted checkpoints.
    if not transformer_sd and any(k.startswith("transformer.") for k in sd.keys()):
        transformer_sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}

    if not transformer_sd:
        raise RuntimeError(
            "Single-file Z-Image checkpoint does not contain transformer weights in expected format."
        )

    missing, unexpected = pipeline.transformer.load_state_dict(transformer_sd, strict=False)
    # Guardrail: if mismatch is huge, fail early rather than silently generating wrong outputs.
    if len(unexpected) > 64:
        raise RuntimeError(
            f"Transformer weight mismatch too large (unexpected={len(unexpected)}). "
            "Checkpoint may be incompatible with selected Z-Image components."
        )
    if len(missing) > 256:
        raise RuntimeError(
            f"Transformer weight mismatch too large (missing={len(missing)}). "
            "Checkpoint may be incompatible with selected Z-Image components."
        )


def resolve_zimage_source(name: str, checkpoint_folders: list[str], auto_download_if_missing: bool = False) -> tuple[Optional[str], Optional[str], str]:
    flavor = detect_zimage_flavor(name)
    resolved = _resolve_named_path(name, checkpoint_folders)

    if resolved is not None:
        if os.path.isdir(resolved) and is_zimage_model_directory(resolved):
            return "directory", resolved, flavor
        if os.path.isfile(resolved):
            # Trust explicit user selection by filename pattern even when key fingerprint
            # is inconclusive (common with repacked/quantized AIO checkpoints).
            if _is_likely_zimage_safetensors(resolved) or is_zimage_checkpoint_name(os.path.basename(resolved)):
                return "single_file", resolved, flavor

    return None, None, flavor


def _pick_device_and_dtype():
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load_pipeline(source_kind: str, source_path: str, flavor: str, checkpoint_folders: list[str]):
    cache_key = f"{source_kind}:{os.path.abspath(source_path)}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    from diffusers import DiffusionPipeline

    device, dtype = _pick_device_and_dtype()

    if source_kind == "directory":
        pipeline = DiffusionPipeline.from_pretrained(
            source_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
    elif source_kind == "single_file":
        if hasattr(DiffusionPipeline, "from_single_file"):
            local_config = _find_local_repo_components(flavor, checkpoint_folders)
            if local_config is None:
                raise RuntimeError(
                    "Single-file Z-Image requires local components (text_encoder/tokenizer/vae/scheduler). "
                    "Provide a local Z-Image model folder in your checkpoint paths."
                )
            pipeline = DiffusionPipeline.from_single_file(
                source_path,
                config=local_config,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            # Backend fallback: build local pipeline from components and inject transformer weights.
            local_config = _find_local_repo_components(flavor, checkpoint_folders)
            if local_config is None:
                raise RuntimeError(
                    "Single-file Z-Image requires local components (text_encoder/tokenizer/vae/scheduler). "
                    "Provide a local Z-Image model folder."
                )
            pipeline = DiffusionPipeline.from_pretrained(
                local_config,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
            _load_transformer_weights_from_single_file(source_path, pipeline)
    else:
        raise ValueError(f"Unsupported source kind: {source_kind}")

    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    _PIPELINE_CACHE[cache_key] = (pipeline, device)
    return _PIPELINE_CACHE[cache_key]


def generate_zimage(
    source_kind: str,
    source_path: str,
    flavor: str,
    checkpoint_folders: list[str],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    import torch

    pipeline, device = _load_pipeline(source_kind, source_path, flavor, checkpoint_folders)
    generator = torch.Generator(device=device).manual_seed(seed)

    call_kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
    )
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    try:
        output = pipeline(**call_kwargs)
    except TypeError:
        call_kwargs.pop("negative_prompt", None)
        output = pipeline(**call_kwargs)

    return output.images[0]
