import json
import os
import hashlib
import importlib
import gc
from typing import Optional


_PIPELINE_CACHE = {}
_PROMPT_EMBED_CACHE = {}
_MAX_PROMPT_CACHE_ITEMS = 32
_TOKENIZER_JSON_SHA256 = {
    "Tongyi-MAI/Z-Image-Turbo": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _clear_prompt_cache_for_pipeline(cache_key: str) -> None:
    stale = [k for k in _PROMPT_EMBED_CACHE.keys() if isinstance(k, tuple) and k and k[0] == cache_key]
    for k in stale:
        _PROMPT_EMBED_CACHE.pop(k, None)


def _put_prompt_cache(key: tuple, value: tuple) -> None:
    if key in _PROMPT_EMBED_CACHE:
        _PROMPT_EMBED_CACHE.pop(key, None)
    _PROMPT_EMBED_CACHE[key] = value
    while len(_PROMPT_EMBED_CACHE) > _MAX_PROMPT_CACHE_ITEMS:
        first = next(iter(_PROMPT_EMBED_CACHE))
        _PROMPT_EMBED_CACHE.pop(first, None)


def _cleanup_memory(cuda: bool = True) -> None:
    gc.collect()
    if not cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _cuda_mem_info_gb() -> tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        return float(free_bytes) / float(1024**3), float(total_bytes) / float(1024**3)
    except Exception:
        return 0.0, 0.0


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


def _is_likely_fp8_single_file(path: str) -> bool:
    if not os.path.isfile(path) or not path.lower().endswith(".safetensors"):
        return False
    name = os.path.basename(path).lower()
    if "fp8" in name:
        return True
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "scaled_fp8" in keys or "transformer.scaled_fp8" in keys:
                return True
            # Common quantized-sidecar naming patterns.
            if any(k.endswith(".scale") or ".fp8_" in k for k in keys):
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


def _split_single_file_state_dict(single_file_path: str, include_aux_weights: bool = False) -> dict[str, dict]:
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

            if include_aux_weights:
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
                        hit = True
                        break
                if hit:
                    continue

            # Plain diffusion-model safetensors often contain bare transformer keys.
            # Keep all non-text/non-vae tensors as transformer weights.
            if any(key.startswith(pref) for pref in text_prefixes):
                if include_aux_weights:
                    text_sd[key] = f.get_tensor(key)
                continue
            if any(key.startswith(pref) for pref in vae_prefixes):
                if include_aux_weights:
                    vae_sd[key] = f.get_tensor(key)
                continue
            transformer_sd[key] = f.get_tensor(key)

    return {
        "transformer": transformer_sd,
        "text_encoder": text_sd,
        "vae": vae_sd,
    }


def _convert_z_image_transformer_checkpoint_to_diffusers(checkpoint: dict) -> dict:
    # Ported from diffusers single-file converter for ZImageTransformer2DModel.
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key
        new_key = new_key.replace("final_layer.", "all_final_layer.2-1.")
        new_key = new_key.replace("x_embedder.", "all_x_embedder.2-1.")
        new_key = new_key.replace(".attention.out.bias", ".attention.to_out.0.bias")
        new_key = new_key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        new_key = new_key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        new_key = new_key.replace(".attention.out.weight", ".attention.to_out.0.weight")
        renamed[new_key] = value

    converted = {}
    for key, value in renamed.items():
        if ".attention.qkv.weight" in key:
            to_q_weight, to_k_weight, to_v_weight = value.chunk(3, dim=0)
            converted[key.replace(".attention.qkv.weight", ".attention.to_q.weight")] = to_q_weight
            converted[key.replace(".attention.qkv.weight", ".attention.to_k.weight")] = to_k_weight
            converted[key.replace(".attention.qkv.weight", ".attention.to_v.weight")] = to_v_weight
            continue
        if ".attention.qkv.bias" in key:
            to_q_bias, to_k_bias, to_v_bias = value.chunk(3, dim=0)
            converted[key.replace(".attention.qkv.bias", ".attention.to_q.bias")] = to_q_bias
            converted[key.replace(".attention.qkv.bias", ".attention.to_k.bias")] = to_k_bias
            converted[key.replace(".attention.qkv.bias", ".attention.to_v.bias")] = to_v_bias
            continue
        converted[key] = value

    return converted


def _convert_z_image_transformer_key_to_diffusers(src_key: str) -> list[str]:
    key = src_key
    key = key.replace("final_layer.", "all_final_layer.2-1.")
    key = key.replace("x_embedder.", "all_x_embedder.2-1.")
    key = key.replace(".attention.out.bias", ".attention.to_out.0.bias")
    key = key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
    key = key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
    key = key.replace(".attention.out.weight", ".attention.to_out.0.weight")
    if ".attention.qkv.weight" in key:
        return [
            key.replace(".attention.qkv.weight", ".attention.to_q.weight"),
            key.replace(".attention.qkv.weight", ".attention.to_k.weight"),
            key.replace(".attention.qkv.weight", ".attention.to_v.weight"),
        ]
    if ".attention.qkv.bias" in key:
        return [
            key.replace(".attention.qkv.bias", ".attention.to_q.bias"),
            key.replace(".attention.qkv.bias", ".attention.to_k.bias"),
            key.replace(".attention.qkv.bias", ".attention.to_v.bias"),
        ]
    return [key]


def _transformer_match_score_from_keys(src_keys, model_keys: set[str]) -> int:
    if not src_keys or not model_keys:
        return 0

    strip_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ]

    matched = 0
    for src_key in src_keys:
        if src_key in model_keys:
            matched += 1
            continue
        for pref in strip_prefixes:
            if src_key.startswith(pref) and src_key[len(pref):] in model_keys:
                matched += 1
                break
    return matched


def _transformer_match_score(state_dict: dict, model_keys: set[str]) -> int:
    return _transformer_match_score_from_keys(state_dict.keys() if state_dict else [], model_keys)


def _load_transformer_weights_from_single_file(single_file_path: str, pipeline) -> None:
    parts = _split_single_file_state_dict(single_file_path, include_aux_weights=True)
    transformer_sd = parts["transformer"]
    transformer_component = getattr(pipeline, "transformer", None)
    transformer_model_keys = (
        set(transformer_component.state_dict().keys()) if transformer_component is not None else set()
    )

    if transformer_sd and transformer_model_keys:
        base_score = _transformer_match_score(transformer_sd, transformer_model_keys)
        try:
            converted_keys = []
            for key in transformer_sd.keys():
                converted_keys.extend(_convert_z_image_transformer_key_to_diffusers(key))
            converted_score = _transformer_match_score_from_keys(converted_keys, transformer_model_keys)
            if converted_score > base_score:
                print(
                    f"[Z-Image POC] Using Forge-style Z-Image transformer key mapping "
                    f"(score {base_score}->{converted_score})."
                )
                transformer_sd = _convert_z_image_transformer_checkpoint_to_diffusers(transformer_sd)
        except Exception as e:
            print(f"[Z-Image POC] Z-Image transformer conversion skipped: {e}")

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
    parts["transformer"].clear()
    parts["text_encoder"].clear()
    parts["vae"].clear()
    del parts
    _cleanup_memory(cuda=False)


def _apply_component_state_dict(
    component,
    state_dict: dict,
    label: str,
    missing_limit: Optional[int] = None,
    unexpected_limit: Optional[int] = None,
):
    if component is None or not state_dict:
        return [], []
    missing, unexpected = component.load_state_dict(state_dict, strict=False)
    if unexpected_limit is not None and len(unexpected) > unexpected_limit:
        raise RuntimeError(f"{label} weight mismatch too large (unexpected={len(unexpected)}).")
    if missing_limit is not None and len(missing) > missing_limit:
        raise RuntimeError(f"{label} weight mismatch too large (missing={len(missing)}).")
    if len(unexpected) > 0 or len(missing) > 0:
        print(f"[Z-Image POC] {label} non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"[Z-Image POC] Loaded {label} from single-file ({len(state_dict)} tensors).")
    return missing, unexpected


def _remap_state_dict_to_model_keys(state_dict: dict, model_keys: set[str], label: str, verbose: bool = True) -> dict:
    if not state_dict:
        return state_dict

    strip_prefixes = [
        "",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ]

    remapped = {}
    matched = 0
    for src_key, value in state_dict.items():
        dst_key = None
        if src_key in model_keys:
            dst_key = src_key
        else:
            for pref in strip_prefixes[1:]:
                if src_key.startswith(pref):
                    cand = src_key[len(pref):]
                    if cand in model_keys:
                        dst_key = cand
                        break

        if dst_key is not None:
            remapped[dst_key] = value
            matched += 1
        else:
            remapped[src_key] = value

    total = max(len(state_dict), 1)
    ratio = matched / float(total)
    if verbose and matched != len(state_dict):
        print(f"[Z-Image POC] {label} remap coverage: matched={matched}/{len(state_dict)} ({ratio:.2%})")
    return remapped


def _call_with_dtype_compat(callable_obj, dtype, kwargs: dict, label: str):
    errors = []

    def _invoke(kwargs_try: dict):
        try:
            return callable_obj(**kwargs_try)
        except TypeError as e:
            # Some loaders reject low_cpu_mem_usage (or dtype args). Retry once without it.
            if "low_cpu_mem_usage" in kwargs_try:
                fallback = dict(kwargs_try)
                fallback.pop("low_cpu_mem_usage", None)
                return callable_obj(**fallback)
            raise e

    if dtype is not None:
        try:
            return _invoke({**kwargs, "dtype": dtype})
        except TypeError as e:
            errors.append(e)
        except Exception:
            raise

        try:
            return _invoke({**kwargs, "torch_dtype": dtype})
        except TypeError as e:
            errors.append(e)
        except Exception:
            raise

    try:
        return _invoke(kwargs)
    except Exception as e:
        if errors:
            print(f"[Z-Image POC] {label} dtype-compat fallback after: {errors[-1]}")
        raise e


def _build_pipeline_from_single_file_components(
    local_config: str,
    single_file_path: str,
    dtype,
    prefer_single_file_aux_weights: bool = False,
):
    from diffusers import DiffusionPipeline
    from transformers import AutoConfig, AutoModel

    model_index = _read_json(os.path.join(local_config, "model_index.json"))
    parts = _split_single_file_state_dict(single_file_path, include_aux_weights=prefer_single_file_aux_weights)

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
                model = _call_with_dtype_compat(
                    cls.from_pretrained,
                    dtype,
                    {
                        "pretrained_model_name_or_path": comp_path,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                    },
                    f"{component_name}.from_pretrained",
                )
        else:
            model = _call_with_dtype_compat(
                cls.from_pretrained,
                dtype,
                {
                    "pretrained_model_name_or_path": comp_path,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                },
                f"{component_name}.from_pretrained",
            )

        if hasattr(model, "to"):
            model = model.to(dtype=dtype)
        components[component_name] = model

    pipeline = pipeline_cls(**components)

    transformer_component = getattr(pipeline, "transformer", None)
    transformer_model_keys = set(transformer_component.state_dict().keys()) if transformer_component is not None else set()
    raw_transformer_sd = parts["transformer"]

    selected_transformer_sd = raw_transformer_sd
    if raw_transformer_sd and transformer_model_keys:
        base_score = _transformer_match_score(raw_transformer_sd, transformer_model_keys)
        try:
            converted_keys = []
            for key in raw_transformer_sd.keys():
                converted_keys.extend(_convert_z_image_transformer_key_to_diffusers(key))
            converted_score = _transformer_match_score_from_keys(converted_keys, transformer_model_keys)
            if converted_score > base_score:
                print(
                    f"[Z-Image POC] Using Forge-style Z-Image transformer key mapping "
                    f"(score {base_score}->{converted_score})."
                )
                selected_transformer_sd = _convert_z_image_transformer_checkpoint_to_diffusers(raw_transformer_sd)
        except Exception as e:
            print(f"[Z-Image POC] Z-Image transformer conversion skipped: {e}")

    remapped_transformer_sd = _remap_state_dict_to_model_keys(
        selected_transformer_sd, transformer_model_keys, "transformer"
    )

    _apply_component_state_dict(
        transformer_component,
        remapped_transformer_sd,
        label="transformer",
        missing_limit=None,
        unexpected_limit=None,
    )
    if prefer_single_file_aux_weights:
        _apply_component_state_dict(getattr(pipeline, "text_encoder", None), parts["text_encoder"], label="text_encoder")
        _apply_component_state_dict(getattr(pipeline, "vae", None), parts["vae"], label="vae")

    if not parts["transformer"]:
        raise RuntimeError("Single-file Z-Image checkpoint does not contain transformer weights in expected format.")

    if isinstance(pipeline, DiffusionPipeline):
        pipeline.set_progress_bar_config(disable=True)

    parts["transformer"].clear()
    parts["text_encoder"].clear()
    parts["vae"].clear()
    del parts
    _cleanup_memory(cuda=False)
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


def _pipeline_has_meta_tensors(pipeline) -> bool:
    for name in ("transformer", "text_encoder", "vae"):
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        has_offload_hook = bool(getattr(module, "_hf_hook", None))
        if not has_offload_hook:
            try:
                has_offload_hook = any(bool(getattr(m, "_hf_hook", None)) for m in module.modules())
            except Exception:
                has_offload_hook = False
        try:
            for p in module.parameters():
                if getattr(p, "is_meta", False):
                    # With accelerate/diffusers offload hooks, meta tensors are expected.
                    # Treat only unmanaged meta tensors as corrupted.
                    if not has_offload_hook:
                        return True
        except Exception:
            continue
    return False


def _cuda_total_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return float(props.total_memory) / float(1024**3)
    except Exception:
        return 0.0


def _choose_memory_mode(device: str) -> tuple[str, float, float, float]:
    import torch

    if device != "cuda":
        return "full_gpu", 0.0, 0.0, 0.0

    total_vram_gb = _cuda_total_vram_gb()
    free_vram_gb = 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        free_vram_gb = float(free_bytes) / float(1024**3)
        if total_vram_gb <= 0:
            total_vram_gb = float(total_bytes) / float(1024**3)
    except Exception:
        pass

    pressure = (free_vram_gb / total_vram_gb) if total_vram_gb > 0 else 0.0

    # Keep this general for consumer GPUs:
    # - <=12GB: sequential offload (8GB/10GB/12GB cards)
    # - <=16GB: model offload (16GB cards)
    # - >16GB: dynamic by free-memory pressure
    if total_vram_gb > 0 and total_vram_gb <= 12.0:
        return "sequential_offload", total_vram_gb, free_vram_gb, pressure
    if total_vram_gb > 0 and total_vram_gb <= 16.0:
        return "model_offload", total_vram_gb, free_vram_gb, pressure
    if pressure < 0.15:
        return "sequential_offload", total_vram_gb, free_vram_gb, pressure
    if pressure < 0.35:
        return "model_offload", total_vram_gb, free_vram_gb, pressure
    return "full_gpu", total_vram_gb, free_vram_gb, pressure


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

        current_mode = getattr(pipeline, "_zimage_memory_mode", "unset")
        target_mode, total_vram_gb, free_vram_gb, pressure = _choose_memory_mode(device)

        # Do not relax memory mode for a cached pipeline:
        # keep the strictest mode already proven stable.
        if current_mode == "sequential_offload":
            target_mode = "sequential_offload"
        elif current_mode == "model_offload" and target_mode == "full_gpu":
            target_mode = "model_offload"

        if target_mode == "sequential_offload" and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            pipeline._zimage_memory_mode = "sequential_offload"
            used_offload = True
            print(
                f"[Z-Image POC] Using sequential CPU offload "
                f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f})."
            )
        elif target_mode in ("model_offload", "sequential_offload") and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            pipeline._zimage_memory_mode = "model_offload"
            used_offload = True
            print(
                f"[Z-Image POC] Using model CPU offload "
                f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f})."
            )
        else:
            if current_mode in ("sequential_offload", "model_offload"):
                # Keep existing offload hooks instead of trying to force full-GPU.
                used_offload = True
            else:
                pipeline.to(device)
                pipeline._zimage_memory_mode = "full_gpu"
            print(
                f"[Z-Image POC] Using full-GPU mode "
                f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f})."
            )
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
        pipeline, _, _ = _PIPELINE_CACHE[cache_key]
        if _pipeline_has_meta_tensors(pipeline):
            print("[Z-Image POC] Cached pipeline has meta tensors, rebuilding pipeline.")
            _PIPELINE_CACHE.pop(cache_key, None)
            _clear_prompt_cache_for_pipeline(cache_key)
            return _load_pipeline(source_kind, source_path, flavor, checkpoint_folders)
        device, _ = _pick_device_and_dtype()
        generator_device, used_offload = _prepare_pipeline_memory_mode(pipeline, device)
        _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
        return _PIPELINE_CACHE[cache_key]

    from diffusers import DiffusionPipeline

    _ensure_zimage_runtime_compatibility()
    prefer_single_file_aux_weights = os.environ.get("FOOOCUS_ZIMAGE_LOAD_AIO_AUX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    device, dtype = _pick_device_and_dtype()

    if source_kind == "directory":
        pipeline = _call_with_dtype_compat(
            DiffusionPipeline.from_pretrained,
            dtype,
            {
                "pretrained_model_name_or_path": source_path,
                "local_files_only": True,
                "low_cpu_mem_usage": True,
            },
            "DiffusionPipeline.from_pretrained(directory)",
        )
    elif source_kind == "single_file":
        local_config, tried_config_only_text_encoder = _ensure_single_file_component_dir(
            flavor, checkpoint_folders, source_path
        )

        split_error = None
        native_error = None
        pipeline = None
        is_fp8_single_file = _is_likely_fp8_single_file(source_path)

        # Forge-like priority: let the framework load the single-file checkpoint natively first.
        try:
            if hasattr(DiffusionPipeline, "from_single_file") and not is_fp8_single_file:
                native_kwargs = dict(
                    config=local_config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                pipeline = _call_with_dtype_compat(
                    lambda **kwargs: DiffusionPipeline.from_single_file(source_path, **kwargs),
                    dtype,
                    native_kwargs,
                    "DiffusionPipeline.from_single_file",
                )
                if pipeline is not None and _pipeline_has_meta_tensors(pipeline):
                    raise RuntimeError("native single-file produced meta tensors")
            elif is_fp8_single_file:
                print("[Z-Image POC] FP8 checkpoint detected, skipping native single-file loader to reduce RAM spikes.")
        except Exception as e:
            native_error = e
            print(f"[Z-Image POC] Native single-file loader fallback due to: {e}")
            _cleanup_memory(cuda=True)

        # Fallback: split-loader assembly.
        if pipeline is None:
            try:
                pipeline = _build_pipeline_from_single_file_components(
                    local_config,
                    source_path,
                    dtype,
                    prefer_single_file_aux_weights=prefer_single_file_aux_weights,
                )
                if pipeline is not None and _pipeline_has_meta_tensors(pipeline):
                    raise RuntimeError("split-loader produced meta tensors")
            except Exception as e:
                split_error = e
                print(f"[Z-Image POC] Split-loader fallback due to: {e}")
                _cleanup_memory(cuda=True)

        if pipeline is None and split_error is not None:
            # Legacy fallback path.
            try:
                pipeline = _call_with_dtype_compat(
                    DiffusionPipeline.from_pretrained,
                    dtype,
                    {
                        "pretrained_model_name_or_path": local_config,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                    },
                    "DiffusionPipeline.from_pretrained(local_config)",
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
                    pipeline = _call_with_dtype_compat(
                        DiffusionPipeline.from_pretrained,
                        dtype,
                        {
                            "pretrained_model_name_or_path": local_config,
                            "local_files_only": True,
                            "low_cpu_mem_usage": True,
                        },
                        "DiffusionPipeline.from_pretrained(local_config-fallback)",
                    )
                except Exception:
                    raise split_error
            _load_transformer_weights_from_single_file(source_path, pipeline)
        if pipeline is None:
            if split_error is not None:
                raise split_error
            if native_error is not None:
                raise native_error
            raise RuntimeError("Failed to build Z-Image pipeline from single-file checkpoint.")
    else:
        raise ValueError(f"Unsupported source kind: {source_kind}")

    if _pipeline_has_meta_tensors(pipeline):
        raise RuntimeError("Z-Image pipeline contains meta tensors after load.")

    pipeline.set_progress_bar_config(disable=True)
    generator_device, used_offload = _prepare_pipeline_memory_mode(pipeline, device)
    _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
    return _PIPELINE_CACHE[cache_key]


def _run_pipeline_call(pipeline, call_kwargs: dict):
    kwargs = dict(call_kwargs)
    optional_drop_order = [
        "cfg_normalization",
        "cfg_truncation",
        "max_sequence_length",
        "negative_prompt",
    ]
    for _ in range(len(optional_drop_order) + 1):
        try:
            return pipeline(**kwargs)
        except TypeError:
            dropped = False
            for key in optional_drop_order:
                if key in kwargs:
                    kwargs.pop(key, None)
                    dropped = True
                    break
            if not dropped:
                raise


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
    shift: float = 3.0,
):
    import torch

    cache_key = f"{source_kind}:{os.path.abspath(source_path)}"
    pipeline, generator_device, used_offload = _load_pipeline(source_kind, source_path, flavor, checkpoint_folders)
    if generator_device == "cuda":
        _cleanup_memory(cuda=True)
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    # Align scheduler shift with Forge-style "Shift" control when available.
    try:
        if hasattr(pipeline, "scheduler") and hasattr(pipeline.scheduler, "config"):
            if hasattr(pipeline.scheduler.config, "shift"):
                pipeline.scheduler.config.shift = float(shift)
            if hasattr(pipeline.scheduler, "shift"):
                pipeline.scheduler.shift = float(shift)
    except Exception:
        pass

    max_sequence_length = 256 if flavor == "turbo" else 512
    forced_max_seq = getattr(pipeline, "_zimage_forced_max_sequence_length", None)
    if forced_max_seq is not None:
        max_sequence_length = min(max_sequence_length, int(forced_max_seq))
    if generator_device == "cuda":
        free_gb, total_gb = _cuda_mem_info_gb()
        if max_sequence_length > 192 and free_gb > 0 and free_gb < 0.40:
            max_sequence_length = 192
            pipeline._zimage_forced_max_sequence_length = 192
            print(
                f"[Z-Image POC] Low free VRAM before generation ({free_gb:.2f}GB/{total_gb:.2f}GB), "
                "using max_sequence_length=192."
            )
        if max_sequence_length > 160 and free_gb > 0 and free_gb < 0.25:
            max_sequence_length = 160
            pipeline._zimage_forced_max_sequence_length = 160
            print(
                f"[Z-Image POC] Very low free VRAM before generation ({free_gb:.2f}GB/{total_gb:.2f}GB), "
                "using max_sequence_length=160."
            )

    use_cfg = guidance_scale > 1.0
    neg_key = negative_prompt if use_cfg else ""
    embed_cache_key = (
        cache_key,
        prompt,
        neg_key,
        int(max_sequence_length),
        bool(use_cfg),
    )

    prompt_embeds = None
    negative_prompt_embeds = None
    cached_embeds = _PROMPT_EMBED_CACHE.get(embed_cache_key, None)
    if cached_embeds is not None:
        cached_pos, cached_neg = cached_embeds
        prompt_embeds = [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in cached_pos]
        if cached_neg:
            negative_prompt_embeds = [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in cached_neg]
        else:
            negative_prompt_embeds = []
    else:
        pos, neg = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if use_cfg else None,
            do_classifier_free_guidance=use_cfg,
            device=generator_device,
            max_sequence_length=max_sequence_length,
        )
        cpu_pos = [x.detach().to("cpu", copy=True) for x in pos]
        cpu_neg = [x.detach().to("cpu", copy=True) for x in neg] if neg else []
        _put_prompt_cache(embed_cache_key, (cpu_pos, cpu_neg))
        prompt_embeds = [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in cpu_pos]
        if cpu_neg:
            negative_prompt_embeds = [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in cpu_neg]
        else:
            negative_prompt_embeds = []

    call_kwargs = dict(
        prompt=None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
        cfg_normalization=False,
        cfg_truncation=1.0,
        max_sequence_length=max_sequence_length,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    print(
        f"[Z-Image POC] Runtime params: steps={steps}, guidance={guidance_scale}, shift={shift}, "
        f"max_seq={max_sequence_length}, offload={used_offload}"
    )

    output = None
    try:
        retry_caps = []
        if flavor == "turbo":
            current_seq = int(call_kwargs.get("max_sequence_length", 256))
            for candidate in (192, 160, 128):
                if current_seq > candidate:
                    retry_caps.append(candidate)

        for attempt in range(3):
            try:
                output = _run_pipeline_call(pipeline, call_kwargs)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" not in msg or generator_device != "cuda":
                    raise
                if attempt >= 2:
                    raise

                print("[Z-Image POC] CUDA OOM detected, retrying with stricter offload mode.")
                _cleanup_memory(cuda=True)

                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()
                    pipeline._zimage_memory_mode = "sequential_offload"
                    used_offload = True
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing("max")
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()

                if retry_caps:
                    next_cap = retry_caps.pop(0)
                    call_kwargs["max_sequence_length"] = next_cap
                    pipeline._zimage_forced_max_sequence_length = next_cap
                    print(
                        f"[Z-Image POC] Retrying with reduced max_sequence_length={next_cap} for lower VRAM usage."
                    )
                else:
                    print("[Z-Image POC] Retrying with same sequence length after memory cleanup.")
                continue

        if output is None:
            raise RuntimeError("Z-Image generation failed after OOM retries.")

        image = output.images[0]
        del output
        return image
    except Exception:
        # Prevent poisoned/corrupted cache from breaking next generation request.
        _PIPELINE_CACHE.pop(cache_key, None)
        _clear_prompt_cache_for_pipeline(cache_key)
        raise
    finally:
        try:
            # Ensure accelerate offload hooks release device-resident weights between images.
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
            del prompt_embeds
            del negative_prompt_embeds
            del generator
            call_kwargs.clear()
        except Exception:
            pass
        _cleanup_memory(cuda=(generator_device == "cuda"))
