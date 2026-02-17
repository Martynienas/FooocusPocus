import json
import os
import hashlib
import importlib
from typing import Optional


_PIPELINE_CACHE = {}
_TOKENIZER_JSON_SHA256 = {
    "Tongyi-MAI/Z-Image-Turbo": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _universal_zimage_root() -> str:
    return os.path.join(_project_root(), "models", "zimage")


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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _single_file_has_text_encoder_weights(path: str) -> bool:
    if not os.path.isfile(path) or not path.lower().endswith(".safetensors"):
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("text_encoders.qwen3_4b.") or key.startswith("text_encoder."):
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
    universal_root = _universal_zimage_root()
    universal_repo_dir = os.path.join(universal_root, repo)

    # Preferred universal location:
    #   models/zimage/<RepoName>/{text_encoder,tokenizer,vae,scheduler}
    if is_zimage_model_directory(universal_repo_dir):
        return universal_repo_dir
    # Backward-compatible fallback:
    #   models/zimage/{text_encoder,tokenizer,vae,scheduler}
    if is_zimage_model_directory(universal_root):
        return universal_root

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


def _download_repo_components(repo_id: str, local_config: str, patterns: list[str], missing: list[str]) -> None:
    from huggingface_hub import snapshot_download

    print(f"[Z-Image POC] Downloading missing components: {', '.join(missing)}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_config,
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )


def _has_local_transformer_weights(local_config: str) -> bool:
    transformer_dir = os.path.join(local_config, "transformer")
    if not os.path.isdir(transformer_dir):
        return False
    for name in os.listdir(transformer_dir):
        lowered = name.lower()
        if lowered.startswith("diffusion_pytorch_model"):
            return True
    return False


def _ensure_single_file_component_dir(
    flavor: str,
    checkpoint_folders: list[str],
    single_file_path: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Prepare a local component directory for single-file Z-Image loading.
    Policy:
    - tokenizer + text_encoder + vae + scheduler may be auto-downloaded if missing.
    - when single-file already includes text-encoder weights, prefer config-only text_encoder bootstrap.
    """
    local_config = _find_local_repo_components(flavor, checkpoint_folders)
    repo_id = _repo_for_flavor(flavor)
    repo = repo_id.split("/")[-1]
    universal_root = _universal_zimage_root()
    preferred_local_config = os.path.join(universal_root, repo)
    os.makedirs(preferred_local_config, exist_ok=True)

    if local_config is None:
        # Use universal repo folder as canonical storage even before it is complete.
        local_config = preferred_local_config

    need_tokenizer = not os.path.isdir(os.path.join(local_config, "tokenizer"))
    need_text_encoder = not os.path.isdir(os.path.join(local_config, "text_encoder"))
    need_vae = not os.path.isdir(os.path.join(local_config, "vae"))
    need_scheduler = not os.path.isdir(os.path.join(local_config, "scheduler"))

    tokenizer_json = os.path.join(local_config, "tokenizer", "tokenizer.json")
    expected_tokenizer_sha = _TOKENIZER_JSON_SHA256.get(repo_id)
    if (not need_tokenizer) and expected_tokenizer_sha and os.path.isfile(tokenizer_json):
        try:
            current_sha = _sha256_file(tokenizer_json)
            if current_sha.lower() != expected_tokenizer_sha.lower():
                print("[Z-Image POC] Local tokenizer checksum mismatch, refreshing tokenizer files.")
                need_tokenizer = True
        except Exception as e:
            print(f"[Z-Image POC] Tokenizer checksum check failed, will refresh tokenizer files: {e}")
            need_tokenizer = True

    if not need_tokenizer and not need_text_encoder and not need_vae and not need_scheduler:
        return local_config, False

    try_config_only_text_encoder = bool(
        need_text_encoder and single_file_path and _single_file_has_text_encoder_weights(single_file_path)
    )

    patterns = [
        "model_index.json",
        "transformer/config.json",
    ]
    if need_tokenizer:
        patterns.append("tokenizer/*")
    if need_text_encoder:
        if try_config_only_text_encoder:
            patterns.append("text_encoder/*.json")
        else:
            patterns.append("text_encoder/*")
    if need_vae:
        patterns.append("vae/*")
    if need_scheduler:
        patterns.append("scheduler/*")

    missing = []
    if need_tokenizer:
        missing.append("tokenizer")
    if need_text_encoder:
        if try_config_only_text_encoder:
            missing.append("text_encoder(config)")
        else:
            missing.append("text_encoder")
    if need_vae:
        missing.append("vae")
    if need_scheduler:
        missing.append("scheduler")
    _download_repo_components(repo_id, local_config, patterns, missing)
    return local_config, try_config_only_text_encoder


def _extract_prefixed_state_dict(sd: dict, prefixes: list[str]) -> dict:
    for pref in prefixes:
        part = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if part:
            return part
    return {}


def _split_single_file_state_dict(single_file_path: str) -> dict[str, dict]:
    from safetensors import safe_open

    transformer_prefixes = ["model.diffusion_model.", "diffusion_model.", "transformer."]
    text_prefixes = ["text_encoders.qwen3_4b.", "text_encoder.", "qwen3_4b.transformer.", "qwen3_4b."]
    vae_prefixes = ["vae.", "first_stage_model."]

    transformer_sd = {}
    text_sd = {}
    vae_sd = {}

    with safe_open(single_file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            hit = False
            for pref in transformer_prefixes:
                if key.startswith(pref):
                    transformer_sd[key[len(pref):]] = f.get_tensor(key)
                    hit = True
                    break
            if hit:
                continue

            for pref in text_prefixes:
                if key.startswith(pref):
                    text_sd[key[len(pref):]] = f.get_tensor(key)
                    hit = True
                    break
            if hit:
                continue

            for pref in vae_prefixes:
                if key.startswith(pref):
                    vae_sd[key[len(pref):]] = f.get_tensor(key)
                    break

    return {
        "transformer": transformer_sd,
        "text_encoder": text_sd,
        "vae": vae_sd,
    }


def _load_transformer_weights_from_single_file(single_file_path: str, pipeline) -> None:
    parts = _split_single_file_state_dict(single_file_path)
    transformer_sd = parts["transformer"]

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

    # Optional: if single-file includes text encoder / VAE weights, prefer them.
    if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
        text_sd = parts["text_encoder"]
        if text_sd:
            try:
                pipeline.text_encoder.load_state_dict(text_sd, strict=False)
                print(f"[Z-Image POC] Loaded text_encoder weights from single-file ({len(text_sd)} tensors).")
            except Exception as e:
                print(f"[Z-Image POC] Skipped text_encoder weights from single-file: {e}")

    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        vae_sd = parts["vae"]
        if vae_sd:
            try:
                pipeline.vae.load_state_dict(vae_sd, strict=False)
                print(f"[Z-Image POC] Loaded VAE weights from single-file ({len(vae_sd)} tensors).")
            except Exception as e:
                print(f"[Z-Image POC] Skipped VAE weights from single-file: {e}")


def _apply_component_state_dict(
    component,
    state_dict: dict,
    label: str,
    missing_limit: Optional[int] = None,
    unexpected_limit: Optional[int] = None,
):
    if component is None or not state_dict:
        return
    missing, unexpected = component.load_state_dict(state_dict, strict=False)
    if unexpected_limit is not None and len(unexpected) > unexpected_limit:
        raise RuntimeError(f"{label} weight mismatch too large (unexpected={len(unexpected)}).")
    if missing_limit is not None and len(missing) > missing_limit:
        raise RuntimeError(f"{label} weight mismatch too large (missing={len(missing)}).")
    if len(unexpected) > 0 or len(missing) > 0:
        print(f"[Z-Image POC] {label} non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"[Z-Image POC] Loaded {label} from single-file ({len(state_dict)} tensors).")


def _build_pipeline_from_single_file_components(local_config: str, single_file_path: str, torch_dtype):
    from diffusers import DiffusionPipeline
    from transformers import AutoConfig, AutoModel

    model_index = _read_json(os.path.join(local_config, "model_index.json"))
    parts = _split_single_file_state_dict(single_file_path)

    pipeline_class_name = str(model_index.get("_class_name", ""))
    pipeline_cls = getattr(importlib.import_module("diffusers"), pipeline_class_name, None)
    if pipeline_cls is None:
        raise RuntimeError(
            f"diffusers is missing {pipeline_class_name}. "
            "Install/upgrade with: python -m pip install -U diffusers==0.36.0 transformers==4.56.2 safetensors accelerate"
        )

    components = {}
    for component_name, spec in model_index.items():
        if component_name.startswith("_"):
            continue
        if not (isinstance(spec, list) and len(spec) == 2):
            continue

        lib_name, cls_name = spec
        comp_path = os.path.join(local_config, component_name)
        lib = importlib.import_module(lib_name)
        cls = getattr(lib, cls_name)

        if component_name == "scheduler":
            components[component_name] = cls.from_pretrained(comp_path, local_files_only=True)
            continue
        if component_name.startswith("tokenizer"):
            components[component_name] = cls.from_pretrained(comp_path, local_files_only=True)
            continue

        want_single_file_weights = bool(parts.get(component_name))

        if component_name == "text_encoder" and want_single_file_weights:
            config = AutoConfig.from_pretrained(comp_path, local_files_only=True, trust_remote_code=True)
            model = AutoModel.from_config(config, trust_remote_code=True)
        elif want_single_file_weights:
            if hasattr(cls, "load_config") and hasattr(cls, "from_config"):
                model = cls.from_config(cls.load_config(comp_path))
            else:
                model = cls.from_pretrained(comp_path, local_files_only=True, torch_dtype=torch_dtype)
        else:
            model = cls.from_pretrained(comp_path, local_files_only=True, torch_dtype=torch_dtype)

        if hasattr(model, "to"):
            model = model.to(dtype=torch_dtype)
        components[component_name] = model

    pipeline = pipeline_cls(**components)

    _apply_component_state_dict(
        getattr(pipeline, "transformer", None),
        parts["transformer"],
        label="transformer",
        missing_limit=None,
        unexpected_limit=None,
    )
    _apply_component_state_dict(getattr(pipeline, "text_encoder", None), parts["text_encoder"], label="text_encoder")
    _apply_component_state_dict(getattr(pipeline, "vae", None), parts["vae"], label="vae")

    if not parts["transformer"]:
        raise RuntimeError("Single-file Z-Image checkpoint does not contain transformer weights in expected format.")

    if isinstance(pipeline, DiffusionPipeline):
        pipeline.set_progress_bar_config(disable=True)

    del parts
    return pipeline


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


def _cuda_total_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return float(props.total_memory) / float(1024**3)
    except Exception:
        return 0.0


def _prepare_pipeline_memory_mode(pipeline, device: str) -> tuple[str, bool]:
    """
    Returns (generator_device, used_offload_mode).
    generator_device is used to seed torch.Generator.
    """
    import torch

    used_offload = False

    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("max")
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        vram_gb = _cuda_total_vram_gb()
        if vram_gb > 0 and vram_gb <= 12.6 and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            used_offload = True
            print(f"[Z-Image POC] Using model CPU offload mode for {vram_gb:.2f} GB VRAM.")
        else:
            pipeline.to(device)
    else:
        pipeline.to(device)

    return ("cuda" if device == "cuda" else "cpu"), used_offload


def _ensure_zimage_runtime_compatibility() -> None:
    missing = []
    try:
        import diffusers
    except Exception as e:
        raise RuntimeError(
            f"Z-Image runtime missing diffusers ({e}). "
            "Install/upgrade with: python -m pip install -U diffusers==0.36.0 transformers==4.56.2 safetensors accelerate"
        ) from e

    try:
        import transformers
    except Exception as e:
        raise RuntimeError(
            f"Z-Image runtime missing transformers ({e}). "
            "Install/upgrade with: python -m pip install -U transformers==4.56.2"
        ) from e

    if not hasattr(diffusers, "ZImagePipeline"):
        missing.append("diffusers.ZImagePipeline")
    if not hasattr(diffusers, "ZImageTransformer2DModel"):
        missing.append("diffusers.ZImageTransformer2DModel")
    if not hasattr(transformers, "Qwen3Model"):
        missing.append("transformers.Qwen3Model")

    if missing:
        dv = getattr(diffusers, "__version__", "unknown")
        tv = getattr(transformers, "__version__", "unknown")
        raise RuntimeError(
            "Z-Image backend is too old for this model. Missing: "
            + ", ".join(missing)
            + f". Current versions: diffusers={dv}, transformers={tv}. "
            "Install/upgrade with: python -m pip install -U diffusers==0.36.0 transformers==4.56.2 safetensors accelerate"
        )


def _load_pipeline(source_kind: str, source_path: str, flavor: str, checkpoint_folders: list[str]):
    cache_key = f"{source_kind}:{os.path.abspath(source_path)}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    from diffusers import DiffusionPipeline

    _ensure_zimage_runtime_compatibility()

    device, dtype = _pick_device_and_dtype()

    if source_kind == "directory":
        pipeline = DiffusionPipeline.from_pretrained(
            source_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
    elif source_kind == "single_file":
        local_config, tried_config_only_text_encoder = _ensure_single_file_component_dir(
            flavor, checkpoint_folders, source_path
        )

        # Prefer split-loader first to avoid native path loading full local transformer
        # weights (which may not exist and can cause long delays).
        split_error = None
        try:
            pipeline = _build_pipeline_from_single_file_components(local_config, source_path, dtype)
        except Exception as e:
            split_error = e
            print(f"[Z-Image POC] Split-loader fallback due to: {e}")

        # If split-loader fails, use native single-file only when local transformer
        # weights are actually available.
        native_error = None
        if split_error is not None and hasattr(DiffusionPipeline, "from_single_file") and _has_local_transformer_weights(local_config):
            try:
                single_file_kwargs = dict(
                    config=local_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                # Avoid forced upcast for fp8 checkpoints.
                if "fp8" not in os.path.basename(source_path).lower():
                    single_file_kwargs["torch_dtype"] = dtype
                pipeline = DiffusionPipeline.from_single_file(
                    source_path,
                    **single_file_kwargs,
                )
            except Exception as e:
                native_error = e
                print(f"[Z-Image POC] Native single-file loader fallback due to: {e}")

        if split_error is not None and (native_error is not None or not hasattr(DiffusionPipeline, "from_single_file")):
            # Legacy fallback path.
            try:
                pipeline = DiffusionPipeline.from_pretrained(
                    local_config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                if not tried_config_only_text_encoder:
                    raise split_error
                # Fallback: some backends require full text_encoder files even when AIO contains weights.
                repo_id = _repo_for_flavor(flavor)
                _download_repo_components(
                    repo_id,
                    local_config,
                    patterns=["text_encoder/*"],
                    missing=["text_encoder(fallback-full)"],
                )
                try:
                    pipeline = DiffusionPipeline.from_pretrained(
                        local_config,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    raise split_error
            _load_transformer_weights_from_single_file(source_path, pipeline)

        if split_error is not None and native_error is None and hasattr(DiffusionPipeline, "from_single_file") and _has_local_transformer_weights(local_config):
            # Native path succeeded and set pipeline.
            pass
        elif split_error is not None and native_error is None and (not hasattr(DiffusionPipeline, "from_single_file") or not _has_local_transformer_weights(local_config)):
            raise split_error
    else:
        raise ValueError(f"Unsupported source kind: {source_kind}")

    pipeline.set_progress_bar_config(disable=True)
    generator_device, used_offload = _prepare_pipeline_memory_mode(pipeline, device)
    _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
    return _PIPELINE_CACHE[cache_key]


def _run_pipeline_call(pipeline, call_kwargs: dict):
    try:
        return pipeline(**call_kwargs)
    except TypeError:
        call_kwargs = dict(call_kwargs)
        call_kwargs.pop("negative_prompt", None)
        return pipeline(**call_kwargs)


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

    pipeline, generator_device, used_offload = _load_pipeline(source_kind, source_path, flavor, checkpoint_folders)
    generator = torch.Generator(device=generator_device).manual_seed(seed)

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
        output = _run_pipeline_call(pipeline, call_kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" not in msg or generator_device != "cuda":
            raise

        print("[Z-Image POC] CUDA OOM detected, retrying with stricter offload mode.")
        torch.cuda.empty_cache()
        if hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            used_offload = True
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("max")
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        output = _run_pipeline_call(pipeline, call_kwargs)

    return output.images[0]
