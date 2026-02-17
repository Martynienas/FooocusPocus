import json
import os
import hashlib
import importlib
import glob
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


def _universal_zimage_root() -> str:
    return os.path.join(_project_root(), "models", "zimage")


def _custom_text_encoder_root() -> str:
    return os.path.join(_project_root(), "models", "text_encoders")


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


def _find_custom_text_encoder_source(flavor: str) -> tuple[Optional[str], Optional[str]]:
    """
    Custom text encoder lookup location for users:
      models/text_encoders/
    Supports:
      - HF-style folder (contains config.json + weights)
      - single safetensors file (loaded with local config bootstrap)
    Returns: (path, kind) where kind is "directory" or "file".
    """
    root = _custom_text_encoder_root()
    if not os.path.isdir(root):
        return None, None

    repo = _repo_for_flavor(flavor).split("/")[-1].lower()
    ordered_names = [
        repo,
        repo.replace("-", "_"),
        "z-image-turbo",
        "z_image_turbo",
        "zimage_turbo",
        "z-image",
        "z_image",
        "zimage",
        "qwen3_4b",
        "qwen3",
        "text_encoder",
    ]

    def _weights_in_dir(path: str) -> bool:
        patterns = (
            "model.safetensors",
            "model-*.safetensors",
            "diffusion_pytorch_model.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "*.safetensors",
            "*.bin",
        )
        for pattern in patterns:
            if glob.glob(os.path.join(path, pattern)):
                return True
        return False

    # Prefer named directories first.
    for name in ordered_names:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            if _weights_in_dir(candidate):
                print(f"[Z-Image POC] Using custom text encoder directory: {candidate}")
                return candidate, "directory"

    # Then any valid directory.
    for entry in sorted(os.listdir(root), key=str.casefold):
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")) and _weights_in_dir(candidate):
            print(f"[Z-Image POC] Using custom text encoder directory: {candidate}")
            return candidate, "directory"

    # Finally single-file safetensors.
    for name in ordered_names:
        candidate = os.path.join(root, f"{name}.safetensors")
        if os.path.isfile(candidate):
            print(f"[Z-Image POC] Using custom text encoder file: {candidate}")
            return candidate, "file"
    for candidate in sorted(glob.glob(os.path.join(root, "*.safetensors")), key=str.casefold):
        print(f"[Z-Image POC] Using custom text encoder file: {candidate}")
        return candidate, "file"

    return None, None


def _ensure_single_file_component_dir(
    flavor: str,
    checkpoint_folders: list[str],
    single_file_path: Optional[str] = None,
) -> tuple[str, bool, Optional[str], Optional[str]]:
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

    custom_text_encoder_path, custom_text_encoder_kind = _find_custom_text_encoder_source(flavor)
    local_text_encoder_dir = os.path.join(local_config, "text_encoder")
    local_text_encoder_present = os.path.isdir(local_text_encoder_dir)
    use_custom_text_encoder_dir = bool(custom_text_encoder_kind == "directory" and custom_text_encoder_path)
    use_custom_text_encoder_file = bool(custom_text_encoder_kind == "file" and custom_text_encoder_path)

    need_tokenizer = not os.path.isdir(os.path.join(local_config, "tokenizer"))
    need_text_encoder = not local_text_encoder_present and not use_custom_text_encoder_dir
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
        return local_config, False, custom_text_encoder_path, custom_text_encoder_kind

    try_config_only_text_encoder = bool(
        need_text_encoder and single_file_path and _single_file_has_text_encoder_weights(single_file_path)
    )
    if need_text_encoder and use_custom_text_encoder_file:
        # We can bootstrap model from config only and load custom safetensors from models/text_encoders.
        try_config_only_text_encoder = True

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
    return local_config, try_config_only_text_encoder, custom_text_encoder_path, custom_text_encoder_kind


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


def _transformer_match_score(state_dict: dict, model_keys: set[str]) -> int:
    if not state_dict or not model_keys:
        return 0

    strip_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ]

    matched = 0
    for src_key in state_dict.keys():
        if src_key in model_keys:
            matched += 1
            continue
        for pref in strip_prefixes:
            if src_key.startswith(pref) and src_key[len(pref) :] in model_keys:
                matched += 1
                break
    return matched


def _load_transformer_weights_from_single_file(single_file_path: str, pipeline) -> None:
    parts = _split_single_file_state_dict(single_file_path, include_aux_weights=True)
    transformer_sd = parts["transformer"]
    transformer_component = getattr(pipeline, "transformer", None)
    transformer_model_keys = (
        set(transformer_component.state_dict().keys()) if transformer_component is not None else set()
    )

    if transformer_sd and transformer_model_keys:
        best_sd = transformer_sd
        base_score = _transformer_match_score(best_sd, transformer_model_keys)
        try:
            converted_sd = _convert_z_image_transformer_checkpoint_to_diffusers(dict(transformer_sd))
            converted_score = _transformer_match_score(converted_sd, transformer_model_keys)
            if converted_score > base_score:
                print(
                    f"[Z-Image POC] Using Forge-style Z-Image transformer key mapping "
                    f"(score {base_score}->{converted_score})."
                )
                transformer_sd = converted_sd
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


def _custom_state_dict_compatibility(state_dict: dict, model_state_dict: dict) -> dict[str, float]:
    model_keys = set(model_state_dict.keys())
    total_input = len(state_dict)
    total_model = len(model_keys)
    if total_input <= 0 or total_model <= 0:
        return {
            "matched": 0.0,
            "shape_mismatch": 0.0,
            "unexpected": float(total_input),
            "match_ratio_input": 0.0,
            "coverage_ratio_model": 0.0,
        }

    matched = 0
    shape_mismatch = 0
    unexpected = 0
    for key, tensor in state_dict.items():
        target = model_state_dict.get(key)
        if target is None:
            unexpected += 1
            continue
        if tuple(getattr(tensor, "shape", ())) == tuple(getattr(target, "shape", ())):
            matched += 1
        else:
            shape_mismatch += 1

    return {
        "matched": float(matched),
        "shape_mismatch": float(shape_mismatch),
        "unexpected": float(unexpected),
        "match_ratio_input": float(matched) / float(max(total_input, 1)),
        "coverage_ratio_model": float(matched) / float(max(total_model, 1)),
    }


def _validate_custom_text_encoder_state_dict(model, state_dict: dict, source_path: str) -> None:
    model_sd = model.state_dict()
    stats = _custom_state_dict_compatibility(state_dict, model_sd)
    matched = int(stats["matched"])
    shape_mismatch = int(stats["shape_mismatch"])
    unexpected = int(stats["unexpected"])
    match_ratio_input = stats["match_ratio_input"]
    coverage_ratio_model = stats["coverage_ratio_model"]
    total_input = len(state_dict)
    total_model = len(model_sd)

    print(
        "[Z-Image POC] custom text_encoder compatibility: "
        f"matched={matched}/{total_input} ({match_ratio_input:.2%}), "
        f"coverage={matched}/{total_model} ({coverage_ratio_model:.2%}), "
        f"shape_mismatch={shape_mismatch}, unexpected={unexpected}"
    )

    # Reject low-overlap checkpoints to avoid silent degraded outputs.
    min_matched = min(256, max(64, int(total_model * 0.35)))
    if matched < min_matched or match_ratio_input < 0.45 or coverage_ratio_model < 0.30:
        raise RuntimeError(
            "Custom text encoder appears incompatible with Z-Image Turbo. "
            f"Matched {matched}/{total_input} tensors, model coverage {coverage_ratio_model:.2%}. "
            "Use a matching Qwen3/Z-Image text encoder or remove custom file."
        )


def _find_custom_vae_file(vae_dir: str) -> Optional[str]:
    if not os.path.isdir(vae_dir):
        return None
    preferred = os.path.join(vae_dir, "ae.safetensors")
    if os.path.isfile(preferred):
        return preferred
    for cand in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors")), key=str.casefold):
        base = os.path.basename(cand).lower()
        if base not in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
            return cand
    return None


def _load_custom_vae_state_dict(path: str) -> dict:
    from safetensors import safe_open

    prefixes = ["vae.", "first_stage_model.", "first_stage_model.model.", "autoencoder.", "model."]
    state_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            mapped = key
            for pref in prefixes:
                if key.startswith(pref):
                    mapped = key[len(pref):]
                    break
            state_dict[mapped] = f.get_tensor(key)
    return state_dict


def _validate_custom_vae_state_dict(model, state_dict: dict, source_path: str) -> None:
    model_sd = model.state_dict()
    stats = _custom_state_dict_compatibility(state_dict, model_sd)
    matched = int(stats["matched"])
    shape_mismatch = int(stats["shape_mismatch"])
    unexpected = int(stats["unexpected"])
    match_ratio_input = stats["match_ratio_input"]
    coverage_ratio_model = stats["coverage_ratio_model"]
    total_input = len(state_dict)
    total_model = len(model_sd)

    print(
        "[Z-Image POC] custom vae compatibility: "
        f"matched={matched}/{total_input} ({match_ratio_input:.2%}), "
        f"coverage={matched}/{total_model} ({coverage_ratio_model:.2%}), "
        f"shape_mismatch={shape_mismatch}, unexpected={unexpected}"
    )

    min_matched = min(192, max(48, int(total_model * 0.30)))
    if matched < min_matched or match_ratio_input < 0.35 or coverage_ratio_model < 0.25:
        raise RuntimeError(
            "Custom VAE appears incompatible with current Z-Image VAE architecture. "
            f"Matched {matched}/{total_input} tensors, model coverage {coverage_ratio_model:.2%}."
        )


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

    if dtype is not None:
        try:
            return callable_obj(**{**kwargs, "dtype": dtype})
        except TypeError as e:
            errors.append(e)
        except Exception:
            raise

        try:
            return callable_obj(**{**kwargs, "torch_dtype": dtype})
        except TypeError as e:
            errors.append(e)
        except Exception:
            raise

    try:
        return callable_obj(**kwargs)
    except Exception as e:
        if errors:
            print(f"[Z-Image POC] {label} dtype-compat fallback after: {errors[-1]}")
        raise e


def _build_pipeline_from_single_file_components(
    local_config: str,
    single_file_path: str,
    dtype,
    prefer_single_file_aux_weights: bool = False,
    custom_text_encoder_path: Optional[str] = None,
    custom_text_encoder_kind: Optional[str] = None,
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
        if component_name == "text_encoder" and custom_text_encoder_kind == "directory" and custom_text_encoder_path:
            comp_path = custom_text_encoder_path
        custom_vae_file = _find_custom_vae_file(comp_path) if component_name == "vae" else None

        if component_name == "text_encoder" and want_single_file_weights:
            config = AutoConfig.from_pretrained(comp_path, local_files_only=True, trust_remote_code=True)
            model = AutoModel.from_config(config, trust_remote_code=True)
        elif component_name == "vae" and custom_vae_file and not want_single_file_weights:
            if hasattr(cls, "load_config") and hasattr(cls, "from_config"):
                model = cls.from_config(cls.load_config(comp_path))
            else:
                model = _call_with_dtype_compat(
                    cls.from_pretrained,
                    dtype,
                    {"pretrained_model_name_or_path": comp_path, "local_files_only": True},
                    f"{component_name}.from_pretrained",
                )
        elif want_single_file_weights:
            if hasattr(cls, "load_config") and hasattr(cls, "from_config"):
                model = cls.from_config(cls.load_config(comp_path))
            else:
                model = _call_with_dtype_compat(
                    cls.from_pretrained,
                    dtype,
                    {"pretrained_model_name_or_path": comp_path, "local_files_only": True},
                    f"{component_name}.from_pretrained",
                )
        else:
            model = _call_with_dtype_compat(
                cls.from_pretrained,
                dtype,
                {"pretrained_model_name_or_path": comp_path, "local_files_only": True},
                f"{component_name}.from_pretrained",
            )

        if (
            component_name == "text_encoder"
            and not want_single_file_weights
            and custom_text_encoder_kind == "file"
            and custom_text_encoder_path
        ):
            try:
                from safetensors import safe_open

                external_sd = {}
                text_prefixes = ["text_encoders.qwen3_4b.", "text_encoder.", "qwen3_4b.transformer.", "qwen3_4b."]
                with safe_open(custom_text_encoder_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        added = False
                        for pref in text_prefixes:
                            if key.startswith(pref):
                                external_sd[key[len(pref):]] = f.get_tensor(key)
                                added = True
                                break
                        if not added:
                            external_sd[key] = f.get_tensor(key)
                model_keys = set(model.state_dict().keys())
                external_sd = _remap_state_dict_to_model_keys(external_sd, model_keys, "text_encoder(custom)")
                _validate_custom_text_encoder_state_dict(model, external_sd, custom_text_encoder_path)
                _apply_component_state_dict(
                    model,
                    external_sd,
                    label="text_encoder(custom)",
                    missing_limit=max(256, int(len(model_keys) * 0.45)),
                    unexpected_limit=max(256, int(len(model_keys) * 0.45)),
                )
            except Exception as e:
                raise RuntimeError(f"Failed custom text_encoder load from {custom_text_encoder_path}: {e}") from e

        if component_name == "vae" and custom_vae_file and not want_single_file_weights:
            try:
                external_sd = _load_custom_vae_state_dict(custom_vae_file)
                model_keys = set(model.state_dict().keys())
                external_sd = _remap_state_dict_to_model_keys(external_sd, model_keys, "vae(custom)")
                _validate_custom_vae_state_dict(model, external_sd, custom_vae_file)
                _apply_component_state_dict(
                    model,
                    external_sd,
                    label="vae(custom)",
                    missing_limit=max(192, int(len(model_keys) * 0.45)),
                    unexpected_limit=max(192, int(len(model_keys) * 0.45)),
                )
                print(f"[Z-Image POC] Using custom VAE file: {custom_vae_file}")
            except Exception as e:
                raise RuntimeError(f"Failed custom VAE load from {custom_vae_file}: {e}") from e

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
            converted_transformer_sd = _convert_z_image_transformer_checkpoint_to_diffusers(dict(raw_transformer_sd))
            converted_score = _transformer_match_score(converted_transformer_sd, transformer_model_keys)
            if converted_score > base_score:
                print(
                    f"[Z-Image POC] Using Forge-style Z-Image transformer key mapping "
                    f"(score {base_score}->{converted_score})."
                )
                selected_transformer_sd = converted_transformer_sd
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
    # - <=10GB: sequential offload (8GB cards)
    # - <=16GB: model offload (12GB/16GB cards)
    # - >16GB: dynamic by free-memory pressure
    if total_vram_gb > 0 and total_vram_gb <= 10.0:
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

        # Do not downgrade from offload modes to full-gpu once hooks are installed.
        if current_mode in ("sequential_offload", "model_offload") and target_mode == "full_gpu":
            target_mode = current_mode

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
            {"pretrained_model_name_or_path": source_path, "local_files_only": True},
            "DiffusionPipeline.from_pretrained(directory)",
        )
    elif source_kind == "single_file":
        (
            local_config,
            tried_config_only_text_encoder,
            custom_text_encoder_path,
            custom_text_encoder_kind,
        ) = _ensure_single_file_component_dir(
            flavor, checkpoint_folders, source_path
        )

        split_error = None
        native_error = None
        pipeline = None

        # Forge-like priority: let the framework load the single-file checkpoint natively first.
        try:
            if hasattr(DiffusionPipeline, "from_single_file"):
                native_kwargs = dict(
                    config=local_config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                if not _is_likely_fp8_single_file(source_path):
                    pipeline = _call_with_dtype_compat(
                        lambda **kwargs: DiffusionPipeline.from_single_file(source_path, **kwargs),
                        dtype,
                        native_kwargs,
                        "DiffusionPipeline.from_single_file",
                    )
                else:
                    print("[Z-Image POC] FP8 checkpoint detected, preferring native single-file loader.")
                    pipeline = DiffusionPipeline.from_single_file(source_path, **native_kwargs)
                if pipeline is not None and _pipeline_has_meta_tensors(pipeline):
                    raise RuntimeError("native single-file produced meta tensors")
        except Exception as e:
            native_error = e
            print(f"[Z-Image POC] Native single-file loader fallback due to: {e}")

        # Fallback: split-loader assembly.
        if pipeline is None:
            try:
                pipeline = _build_pipeline_from_single_file_components(
                    local_config,
                    source_path,
                    dtype,
                    prefer_single_file_aux_weights=prefer_single_file_aux_weights,
                    custom_text_encoder_path=custom_text_encoder_path,
                    custom_text_encoder_kind=custom_text_encoder_kind,
                )
                if pipeline is not None and _pipeline_has_meta_tensors(pipeline):
                    raise RuntimeError("split-loader produced meta tensors")
            except Exception as e:
                split_error = e
                print(f"[Z-Image POC] Split-loader fallback due to: {e}")

        if pipeline is None and split_error is not None:
            # Legacy fallback path.
            try:
                pipeline = _call_with_dtype_compat(
                    DiffusionPipeline.from_pretrained,
                    dtype,
                    {"pretrained_model_name_or_path": local_config, "local_files_only": True},
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
                        {"pretrained_model_name_or_path": local_config, "local_files_only": True},
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
    shift: float = 9.0,
):
    import torch

    cache_key = f"{source_kind}:{os.path.abspath(source_path)}"
    pipeline, generator_device, used_offload = _load_pipeline(source_kind, source_path, flavor, checkpoint_folders)
    if generator_device == "cuda":
        torch.cuda.empty_cache()
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

    try:
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
                pipeline._zimage_memory_mode = "sequential_offload"
                used_offload = True
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
            if flavor == "turbo" and call_kwargs.get("max_sequence_length", 0) > 192:
                call_kwargs["max_sequence_length"] = 192
                pipeline._zimage_forced_max_sequence_length = 192
                print("[Z-Image POC] Retrying with reduced max_sequence_length=192 for lower VRAM usage.")
            output = _run_pipeline_call(pipeline, call_kwargs)

        image = output.images[0]
        del output
        if generator_device == "cuda":
            torch.cuda.empty_cache()
        return image
    except Exception:
        # Prevent poisoned/corrupted cache from breaking next generation request.
        _PIPELINE_CACHE.pop(cache_key, None)
        _clear_prompt_cache_for_pipeline(cache_key)
        if generator_device == "cuda":
            torch.cuda.empty_cache()
        raise
