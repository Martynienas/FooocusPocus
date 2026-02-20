import json
import os
import hashlib
import importlib
import gc
import re
import time
import ctypes
import ctypes.util
from typing import Optional


_PIPELINE_CACHE = {}
_PROMPT_EMBED_CACHE = {}
_MAX_PROMPT_CACHE_ITEMS = 32
_TRANSFORMER_MAPPING_DECISION_CACHE = {}
_MAX_TRANSFORMER_MAPPING_DECISIONS = 32
_TRANSFORMER_MAPPING_CACHE_VERSION = "v1"
_PERSISTENT_TRANSFORMER_CACHE_VERSION = "v1"
_ENV_WARNING_ONCE = set()
_LAST_ZIMAGE_SOURCE_SIGNATURE = None
_TOKENIZER_JSON_SHA256 = {
    "Tongyi-MAI/Z-Image-Turbo": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
ZIMAGE_COMPONENT_AUTO = "Auto (use model default)"


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pipeline_cache_key(
    source_kind: str,
    source_path: str,
    text_encoder_override: Optional[str] = None,
    vae_override: Optional[str] = None,
) -> str:
    te = os.path.abspath(text_encoder_override) if text_encoder_override else "-"
    vae = os.path.abspath(vae_override) if vae_override else "-"
    return f"{source_kind}:{os.path.abspath(source_path)}:te={te}:vae={vae}"


def _single_file_identity(path: str) -> str:
    abspath = os.path.abspath(path)
    try:
        st = os.stat(abspath)
        return f"{abspath}:{int(st.st_size)}:{int(st.st_mtime_ns)}"
    except OSError:
        return f"{abspath}:missing"


def _keys_signature(keys: set[str]) -> str:
    hasher = hashlib.sha1()
    for key in sorted(keys):
        hasher.update(key.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return f"{len(keys)}:{hasher.hexdigest()}"


def _transformer_mapping_cache_key(single_file_path: str, model_keys: set[str]) -> str:
    return "|".join(
        (
            _TRANSFORMER_MAPPING_CACHE_VERSION,
            _single_file_identity(single_file_path),
            _keys_signature(model_keys),
        )
    )


def _mapping_cache_get(cache_key: str):
    entry = _TRANSFORMER_MAPPING_DECISION_CACHE.pop(cache_key, None)
    if entry is not None:
        # Keep recently used entries warm in insertion order.
        _TRANSFORMER_MAPPING_DECISION_CACHE[cache_key] = entry
    return entry


def _mapping_cache_put(cache_key: str, value: dict) -> None:
    if cache_key in _TRANSFORMER_MAPPING_DECISION_CACHE:
        _TRANSFORMER_MAPPING_DECISION_CACHE.pop(cache_key, None)
    _TRANSFORMER_MAPPING_DECISION_CACHE[cache_key] = value
    while len(_TRANSFORMER_MAPPING_DECISION_CACHE) > _MAX_TRANSFORMER_MAPPING_DECISIONS:
        oldest = next(iter(_TRANSFORMER_MAPPING_DECISION_CACHE))
        _TRANSFORMER_MAPPING_DECISION_CACHE.pop(oldest, None)


def _zimage_persist_converted_cache_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_PERSIST_CONVERTED_CACHE", "1")


def _zimage_persist_converted_cache_dir() -> str:
    raw = os.environ.get("FOOOCUS_ZIMAGE_PERSIST_CONVERTED_CACHE_DIR", "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.join(os.path.expanduser("~"), ".cache", "fooocuspocus", "zimage", "transformer_converted")


def _zimage_persist_converted_max_items() -> int:
    raw = os.environ.get("FOOOCUS_ZIMAGE_PERSIST_CONVERTED_MAX_ITEMS", "").strip()
    if raw == "":
        return 2
    try:
        value = int(raw)
        if value < 1:
            raise ValueError()
        return min(value, 64)
    except Exception:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_PERSIST_CONVERTED_MAX_ITEMS",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_PERSIST_CONVERTED_MAX_ITEMS='{raw}'.",
        )
        return 2


def _persistent_transformer_cache_paths(single_file_path: str) -> tuple[str, str]:
    source_id = _single_file_identity(single_file_path)
    key = f"{_PERSISTENT_TRANSFORMER_CACHE_VERSION}|{source_id}"
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    cache_dir = _zimage_persist_converted_cache_dir()
    return (
        os.path.join(cache_dir, f"{digest}.safetensors"),
        os.path.join(cache_dir, f"{digest}.json"),
    )


def _cleanup_persisted_transformer_cache(cache_dir: str) -> None:
    max_items = _zimage_persist_converted_max_items()
    try:
        names = [n for n in os.listdir(cache_dir) if n.endswith(".safetensors")]
        files = [os.path.join(cache_dir, n) for n in names]
        files = [p for p in files if os.path.isfile(p)]
        if len(files) <= max_items:
            return
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for stale in files[max_items:]:
            meta = os.path.splitext(stale)[0] + ".json"
            try:
                os.remove(stale)
            except OSError:
                pass
            try:
                if os.path.isfile(meta):
                    os.remove(meta)
            except OSError:
                pass
    except Exception:
        pass


def _load_persisted_converted_transformer(single_file_path: str) -> Optional[dict]:
    if not _zimage_persist_converted_cache_enabled():
        return None
    tensor_path, meta_path = _persistent_transformer_cache_paths(single_file_path)
    if not os.path.isfile(tensor_path):
        return None

    try:
        from safetensors.torch import load_file as safetensors_load_file

        state_dict = safetensors_load_file(tensor_path, device="cpu")
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return None
        now = time.time()
        try:
            os.utime(tensor_path, (now, now))
            if os.path.isfile(meta_path):
                os.utime(meta_path, (now, now))
        except OSError:
            pass
        print(
            f"[Z-Image POC] Loaded persisted converted transformer cache "
            f"({len(state_dict)} tensors)."
        )
        return state_dict
    except Exception as e:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_PERSIST_CONVERTED_CACHE",
            f"[Z-Image POC] Failed to load persisted converted transformer cache: {e}",
        )
        try:
            os.remove(tensor_path)
        except OSError:
            pass
        try:
            if os.path.isfile(meta_path):
                os.remove(meta_path)
        except OSError:
            pass
        return None


def _save_persisted_converted_transformer(single_file_path: str, state_dict: dict) -> None:
    if not _zimage_persist_converted_cache_enabled() or not state_dict:
        return
    tensor_path, meta_path = _persistent_transformer_cache_paths(single_file_path)
    if os.path.isfile(tensor_path):
        return

    cache_dir = os.path.dirname(tensor_path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_PERSIST_CONVERTED_CACHE",
            f"[Z-Image POC] Failed to prepare persistent cache dir '{cache_dir}': {e}",
        )
        return

    tensor_tmp = f"{tensor_path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    meta_tmp = f"{meta_path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    try:
        from safetensors.torch import save_file as safetensors_save_file

        safetensors_save_file(state_dict, tensor_tmp)
        os.replace(tensor_tmp, tensor_path)

        meta = {
            "version": _PERSISTENT_TRANSFORMER_CACHE_VERSION,
            "source_identity": _single_file_identity(single_file_path),
            "created_at_unix": int(time.time()),
            "tensor_count": len(state_dict),
        }
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True)
        os.replace(meta_tmp, meta_path)
        _cleanup_persisted_transformer_cache(cache_dir)
        print(
            f"[Z-Image POC] Saved persisted converted transformer cache "
            f"({len(state_dict)} tensors)."
        )
    except Exception as e:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_PERSIST_CONVERTED_CACHE",
            f"[Z-Image POC] Failed to save persisted converted transformer cache: {e}",
        )
        try:
            if os.path.isfile(tensor_tmp):
                os.remove(tensor_tmp)
        except OSError:
            pass
        try:
            if os.path.isfile(meta_tmp):
                os.remove(meta_tmp)
        except OSError:
            pass


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


def _drop_cache_entry(cache_key: str) -> None:
    _PIPELINE_CACHE.pop(cache_key, None)
    _clear_prompt_cache_for_pipeline(cache_key)


def _cleanup_memory(cuda: bool = True, aggressive: bool = True) -> None:
    if aggressive:
        gc.collect()
        _trim_process_heap()
    if not cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _trim_process_heap() -> None:
    # Release glibc free pages back to OS to reduce RSS spikes on model switches.
    if os.name != "posix":
        return
    try:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name)
        trim = getattr(libc, "malloc_trim", None)
        if callable(trim):
            trim(0)
    except Exception:
        pass


def clear_runtime_caches(flush_cuda: bool = True, aggressive: bool = True) -> dict:
    released_pipelines = 0
    prompt_cache_entries = len(_PROMPT_EMBED_CACHE)
    move_to_cpu_on_clear = _truthy_env("FOOOCUS_ZIMAGE_CACHE_CLEAR_MOVE_TO_CPU", "0")

    for _, cached in list(_PIPELINE_CACHE.items()):
        if not isinstance(cached, tuple) or len(cached) < 1:
            continue
        pipeline = cached[0]
        released_pipelines += 1
        try:
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
        except Exception:
            pass
        try:
            remove_all_hooks = getattr(pipeline, "remove_all_hooks", None)
            if callable(remove_all_hooks):
                remove_all_hooks()
        except Exception:
            pass
        # Break strong references from pipeline objects to large modules early.
        for attr in (
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "vae",
            "tokenizer",
            "tokenizer_2",
            "scheduler",
        ):
            try:
                if hasattr(pipeline, attr):
                    setattr(pipeline, attr, None)
            except Exception:
                pass
        # Keep this opt-in: forcing `pipeline.to("cpu")` can create transient RAM spikes
        # with very large text encoders during model switches.
        if move_to_cpu_on_clear:
            try:
                if hasattr(pipeline, "to"):
                    pipeline.to("cpu")
            except Exception:
                # Some offload/meta-backed modules cannot be moved directly; hook cleanup above is enough.
                pass
        try:
            del pipeline
        except Exception:
            pass

    _PIPELINE_CACHE.clear()
    _PROMPT_EMBED_CACHE.clear()
    _cleanup_memory(cuda=flush_cuda, aggressive=aggressive)
    return {"pipelines": released_pipelines, "prompt_cache_entries": prompt_cache_entries}


def _zimage_harsh_cleanup_on_model_change_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_HARSH_CLEANUP_ON_MODEL_CHANGE", "1")


def maybe_cleanup_for_model_change(source_kind: Optional[str], source_path: Optional[str]) -> dict:
    global _LAST_ZIMAGE_SOURCE_SIGNATURE

    if not source_kind or not source_path:
        return {"changed": False, "cleaned": False, "pipelines": 0, "prompt_cache_entries": 0}

    signature = (str(source_kind), os.path.abspath(os.path.realpath(source_path)))
    previous_signature = _LAST_ZIMAGE_SOURCE_SIGNATURE
    _LAST_ZIMAGE_SOURCE_SIGNATURE = signature

    changed = previous_signature is not None and previous_signature != signature
    if not changed:
        return {"changed": False, "cleaned": False, "pipelines": 0, "prompt_cache_entries": 0}

    if not _zimage_harsh_cleanup_on_model_change_enabled():
        return {"changed": True, "cleaned": False, "pipelines": 0, "prompt_cache_entries": 0}

    stats = clear_runtime_caches(flush_cuda=True, aggressive=True)
    prev_name = os.path.basename(previous_signature[1]) if previous_signature else "unknown"
    next_name = os.path.basename(signature[1])
    print(
        "[Z-Image POC] Model source changed; applied harsh runtime cleanup "
        f"({prev_name} -> {next_name}): "
        f"pipelines={stats.get('pipelines', 0)}, prompt_cache_entries={stats.get('prompt_cache_entries', 0)}."
    )
    return {
        "changed": True,
        "cleaned": True,
        "pipelines": int(stats.get("pipelines", 0)),
        "prompt_cache_entries": int(stats.get("prompt_cache_entries", 0)),
    }


def _cuda_mem_info_gb() -> tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        return float(free_bytes) / float(1024**3), float(total_bytes) / float(1024**3)
    except Exception:
        return 0.0, 0.0


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _iter_safetensors_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".safetensors"):
                yield os.path.join(dirpath, filename)


def _zimage_fp16_safety_from_safetensors(path: str) -> Optional[bool]:
    try:
        import torch
        from safetensors import safe_open
    except Exception:
        return None

    layer_key = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn_norm1\.weight$")

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            candidates = []
            for key in f.keys():
                m = layer_key.search(key)
                if m is not None:
                    candidates.append((int(m.group(1)), key))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0])
            # Mirror Neo's heuristic: inspect layer n_layers - 2 (fallback to max layer).
            target_idx = candidates[-1][0] - 1
            selected_key = None
            for idx, key in reversed(candidates):
                if idx == target_idx:
                    selected_key = key
                    break
            if selected_key is None:
                selected_key = candidates[-1][1]

            weight = f.get_tensor(selected_key)
    except Exception:
        return None

    try:
        std = torch.std(weight, unbiased=False).item()
    except Exception:
        return None

    return bool(std < 0.42)


def _detect_zimage_allow_fp16(source_kind: str, source_path: str) -> Optional[bool]:
    explicit = os.environ.get("FOOOCUS_ZIMAGE_ALLOW_FP16", "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if explicit in ("0", "false", "no", "off"):
        return False

    candidates = []
    if source_kind == "single_file" and source_path.endswith(".safetensors"):
        candidates = [source_path]
    elif source_kind == "directory":
        search_root = os.path.join(source_path, "transformer")
        if not os.path.isdir(search_root):
            search_root = source_path

        candidates = sorted(
            _iter_safetensors_files(search_root),
            key=lambda p: (
                0 if "transformer" in p.lower() else 1,
                0 if os.path.basename(p).startswith("diffusion_pytorch_model") else 1,
                -os.path.getsize(p) if os.path.isfile(p) else 0,
            ),
        )

    for candidate in candidates:
        safe = _zimage_fp16_safety_from_safetensors(candidate)
        if safe is not None:
            status = "safe" if safe else "unsafe"
            print(f"[Z-Image POC] Detected fp16 as {status} from: {os.path.basename(candidate)}")
            return safe

    return None


def _zimage_perf_profile() -> str:
    profile = os.environ.get("FOOOCUS_ZIMAGE_PERF_PROFILE", "safe").strip().lower()
    if profile not in ("safe", "balanced", "speed"):
        return "safe"
    return profile


def _zimage_compute_dtype_mode() -> str:
    raw = os.environ.get("FOOOCUS_ZIMAGE_COMPUTE_DTYPE", "").strip().lower()
    if raw == "":
        raw = os.environ.get("FOOOCUS_ZIMAGE_DTYPE", "auto").strip().lower()

    aliases = {
        "auto": "auto",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "full": "fp32",
    }
    mode = aliases.get(raw, None)
    if mode is None:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_COMPUTE_DTYPE",
            f"[Z-Image POC] Ignoring invalid compute dtype '{raw}'. Expected: auto|bf16|fp16|fp32.",
        )
        return "auto"
    return mode


def _zimage_strict_fp16_mode() -> bool:
    return _zimage_compute_dtype_mode() == "fp16"


def _zimage_fp16_quant_accum_mode() -> str:
    raw = os.environ.get("FOOOCUS_ZIMAGE_COMFY_RUNTIME_FP16_ACCUM", "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "full": "fp32",
    }
    mode = aliases.get(raw, None)
    if mode is None:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_COMFY_RUNTIME_FP16_ACCUM",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_COMFY_RUNTIME_FP16_ACCUM='{raw}'. "
            "Expected: auto|fp16|bf16|fp32.",
        )
        return "auto"
    return mode


def _zimage_prewarm_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_PREWARM", "1")


def _zimage_prewarm_steps() -> int:
    raw = os.environ.get("FOOOCUS_ZIMAGE_PREWARM_STEPS", "").strip()
    if raw == "":
        return 1
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_PREWARM_STEPS",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_PREWARM_STEPS='{raw}'.",
        )
        return 1


def _zimage_prewarm_size(default_width: int = 832, default_height: int = 1216) -> tuple[int, int]:
    raw_w = os.environ.get("FOOOCUS_ZIMAGE_PREWARM_WIDTH", "").strip()
    raw_h = os.environ.get("FOOOCUS_ZIMAGE_PREWARM_HEIGHT", "").strip()
    width = default_width
    height = default_height
    if raw_w:
        try:
            width = max(256, int(raw_w))
        except Exception:
            _warn_once_env(
                "FOOOCUS_ZIMAGE_PREWARM_WIDTH",
                f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_PREWARM_WIDTH='{raw_w}'.",
            )
    if raw_h:
        try:
            height = max(256, int(raw_h))
        except Exception:
            _warn_once_env(
                "FOOOCUS_ZIMAGE_PREWARM_HEIGHT",
                f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_PREWARM_HEIGHT='{raw_h}'.",
            )
    width = int(width // 64) * 64
    height = int(height // 64) * 64
    width = max(width, 256)
    height = max(height, 256)
    return width, height


def _zimage_black_image_retry_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_BLACK_IMAGE_RETRY", "1")


def _zimage_black_image_max_value() -> int:
    raw = os.environ.get("FOOOCUS_ZIMAGE_BLACK_IMAGE_MAX_VALUE", "").strip()
    if raw == "":
        return 8
    try:
        value = int(raw)
        if value < 0:
            raise ValueError()
        return min(value, 32)
    except Exception:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_BLACK_IMAGE_MAX_VALUE",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_BLACK_IMAGE_MAX_VALUE='{raw}'.",
        )
        return 8


def _zimage_black_image_mean_threshold() -> float:
    raw = os.environ.get("FOOOCUS_ZIMAGE_BLACK_IMAGE_MEAN_THRESHOLD", "").strip()
    if raw == "":
        return 2.0
    try:
        value = float(raw)
        if value < 0.0:
            raise ValueError()
        return min(value, 10.0)
    except Exception:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_BLACK_IMAGE_MEAN_THRESHOLD",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_BLACK_IMAGE_MEAN_THRESHOLD='{raw}'.",
        )
        return 2.0


def _analyze_black_image(image) -> Optional[dict]:
    try:
        from PIL import ImageStat
    except Exception:
        return None

    try:
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        stats = ImageStat.Stat(rgb)
        max_value = max(ch_max for _, ch_max in extrema)
        mean_value = float(sum(stats.mean) / max(len(stats.mean), 1))
        std_value = float(sum(stats.stddev) / max(len(stats.stddev), 1))
        return {
            "max": float(max_value),
            "mean": mean_value,
            "std": std_value,
        }
    except Exception:
        return None


def _is_suspected_black_image(image) -> tuple[bool, Optional[dict]]:
    info = _analyze_black_image(image)
    if info is None:
        return False, None
    max_cap = float(_zimage_black_image_max_value())
    mean_cap = float(_zimage_black_image_mean_threshold())
    is_black = info["max"] <= max_cap and info["mean"] <= mean_cap
    return is_black, info


def _retune_runtime_quant_modules_dtype(root_module, dtype) -> int:
    if root_module is None:
        return 0
    changed = 0
    try:
        modules_iter = root_module.modules()
    except Exception:
        modules_iter = []
    for module in modules_iter:
        if not hasattr(module, "compute_dtype"):
            continue
        if not hasattr(module, "quant_format"):
            continue
        old_dtype = getattr(module, "compute_dtype", None)
        try:
            module.compute_dtype = dtype
            if old_dtype != dtype:
                changed += 1
        except Exception:
            continue
        clear_cache = getattr(module, "_clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                pass
    return changed


def _warn_once_env(key: str, message: str) -> None:
    token = f"{key}:{message}"
    if token in _ENV_WARNING_ONCE:
        return
    _ENV_WARNING_ONCE.add(token)
    print(message)


def _zimage_forced_memory_mode() -> Optional[str]:
    raw = os.environ.get("FOOOCUS_ZIMAGE_FORCE_MEMORY_MODE", "").strip().lower()
    if raw == "":
        return None

    aliases = {
        "full_gpu": "full_gpu",
        "full": "full_gpu",
        "gpu": "full_gpu",
        "model_offload": "model_offload",
        "model": "model_offload",
        "sequential_offload": "sequential_offload",
        "sequential": "sequential_offload",
    }
    forced = aliases.get(raw, None)
    if forced is None:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_FORCE_MEMORY_MODE",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_FORCE_MEMORY_MODE='{raw}'. "
            "Expected one of: full_gpu, model_offload, sequential_offload.",
        )
        return None

    return forced


def _zimage_stage_timers_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_STAGE_TIMERS", "0")


def _zimage_allow_quality_fallback() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_ALLOW_QUALITY_FALLBACK", "0")


def _zimage_preemptive_cuda_cleanup_enabled() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_PREEMPTIVE_CUDA_CLEANUP", "1")


def _zimage_preemptive_cuda_cleanup_aggressive() -> bool:
    return _truthy_env("FOOOCUS_ZIMAGE_PREEMPTIVE_CUDA_CLEANUP_AGGRESSIVE", "0")


def _zimage_reserved_vram_gb(total_vram_gb: float = 0.0) -> float:
    raw = os.environ.get("FOOOCUS_ZIMAGE_RESERVE_VRAM_GB", "").strip()
    if raw:
        try:
            reserve = max(0.0, float(raw))
            return reserve
        except Exception:
            _warn_once_env(
                "FOOOCUS_ZIMAGE_RESERVE_VRAM_GB",
                f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_RESERVE_VRAM_GB='{raw}'.",
            )

    # Mirror Comfy defaults.
    if os.name == "nt":
        reserve = 0.6
        if total_vram_gb >= 15.0:
            reserve += 0.1
        return reserve
    return 0.4


def _zimage_model_offload_min_gap_gb() -> float:
    raw = os.environ.get("FOOOCUS_ZIMAGE_MODEL_OFFLOAD_MIN_GAP_GB", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except Exception:
            _warn_once_env(
                "FOOOCUS_ZIMAGE_MODEL_OFFLOAD_MIN_GAP_GB",
                f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_MODEL_OFFLOAD_MIN_GAP_GB='{raw}'.",
            )
    # Conservative default for turbo on 10-12GB class cards.
    return 1.8


def _zimage_vram_estimate_scale() -> float:
    raw = os.environ.get("FOOOCUS_ZIMAGE_VRAM_ESTIMATE_SCALE", "").strip()
    if raw == "":
        return 1.0
    try:
        value = float(raw)
        if value <= 0.0:
            raise ValueError()
        return min(value, 4.0)
    except Exception:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_VRAM_ESTIMATE_SCALE",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_VRAM_ESTIMATE_SCALE='{raw}'.",
        )
        return 1.0


def _format_timing_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000.0:.1f}ms"


def _zimage_xformers_mode() -> str:
    value = os.environ.get("FOOOCUS_ZIMAGE_XFORMERS", "on").strip().lower()
    if value in ("1", "true", "yes", "on", "force"):
        return "on"
    if value in ("0", "false", "no", "off", "disable"):
        return "off"
    return "auto"


def _zimage_attention_backend_mode() -> str:
    value = os.environ.get("FOOOCUS_ZIMAGE_ATTN_BACKEND", "auto").strip().lower()
    if value in ("", "auto", "default"):
        return "auto"
    if value in ("flash", "flash2", "flash-attn", "flash_attention", "flash_attention_2", "fa2"):
        return "flash"
    if value in ("sdpa", "torch", "torch_sdpa"):
        return "sdpa"
    if value in ("xformers", "xformer", "xf"):
        return "xformers"
    if value in ("native", "none", "off", "disable"):
        return "native"
    return "auto"


def _zimage_attention_backend_candidates(mode: str, allow_xformers: bool = True) -> list[str]:
    if mode == "native":
        return ["native"]
    if mode == "sdpa":
        return ["flash_attention_2", "flash_attention", "sdpa", "native"]
    if mode == "flash":
        return ["flash_attention_2", "flash_attention", "flash", "sdpa", "native"]
    if mode == "xformers":
        return ["xformers", "native"]

    # auto
    candidates = ["flash_attention_2", "flash_attention", "flash", "sdpa"]
    if allow_xformers:
        candidates.append("xformers")
    candidates.append("native")
    return candidates


def _is_flash_attention_backend(name: str) -> bool:
    lowered = str(name).strip().lower()
    return lowered.startswith("flash")


def _parse_backend_names_from_error(message: str) -> list[str]:
    text = str(message or "")
    m = re.search(r"must be one of the following:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip().splitlines()[0]
    items = []
    for part in raw.split(","):
        item = part.strip().strip("`'\"")
        if item:
            items.append(item)
    return items


def _discover_transformer_attention_backends(transformer) -> list[str]:
    attrs = (
        "get_supported_attention_backends",
        "get_attention_backends",
        "list_attention_backends",
        "available_attention_backends",
        "attention_backends",
    )
    for attr in attrs:
        obj = getattr(transformer, attr, None)
        if obj is None:
            continue
        try:
            value = obj() if callable(obj) else obj
        except Exception:
            continue
        if isinstance(value, dict):
            values = list(value.keys())
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            continue
        result = []
        for v in values:
            s = str(v).strip()
            if s:
                result.append(s)
        if result:
            return result
    return []


def _expand_attention_backend_alias(name: str) -> list[str]:
    key = str(name).strip().lower()
    alias_map = {
        "flash_attention_2": [
            "flash_attention_2",
            "flash",
            "flash_hub",
            "flash_varlen",
            "flash_varlen_hub",
            "_flash_3",
            "_flash_varlen_3",
            "_flash_3_hub",
            "_native_flash",
        ],
        "flash_attention": [
            "flash_attention",
            "flash",
            "flash_hub",
            "flash_varlen",
            "flash_varlen_hub",
            "_native_flash",
        ],
        "flash": [
            "flash",
            "flash_hub",
            "flash_varlen",
            "flash_varlen_hub",
            "_flash_3",
            "_flash_varlen_3",
            "_flash_3_hub",
            "_native_flash",
        ],
        "sdpa": [
            "sdpa",
            "_native_efficient",
            "_native_math",
            "_native_flash",
            "_native_cudnn",
            "native",
        ],
        "native": [
            "native",
            "_native_math",
            "_native_efficient",
            "_native_flash",
            "_native_cudnn",
        ],
        "xformers": ["xformers"],
    }
    expanded = alias_map.get(key, [key])
    return list(dict.fromkeys(expanded))


def _remap_attention_backend_candidates(candidates: list[str], available: list[str]) -> list[str]:
    if not available:
        return candidates
    available_map = {str(x).strip().lower(): str(x).strip() for x in available if str(x).strip()}
    remapped = []
    for candidate in candidates:
        picked = None
        for alias in _expand_attention_backend_alias(candidate):
            if alias in available_map:
                picked = available_map[alias]
                break
        if picked is None:
            picked = candidate
        remapped.append(picked)
    # Keep order while deduping.
    return list(dict.fromkeys(remapped))


def _round_up_to_supported_seq(value: int, max_cap: int) -> int:
    buckets = [32, 64, 96, 128, 160, 192, 256, 384, 512]
    cap = max(32, int(max_cap))
    for bucket in buckets:
        if bucket >= value and bucket <= cap:
            return bucket
    return cap


def _normalize_prompt_text_for_count(text) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (list, tuple)):
        for item in text:
            if isinstance(item, str) and item.strip():
                return item
        if text:
            return str(text[0])
        return ""
    return str(text)


def _count_tokens_from_input_ids(ids) -> int:
    if ids is None:
        return 0

    shape = getattr(ids, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 2:
                return max(0, int(shape[-1]))
            if len(shape) == 1:
                return max(0, int(shape[0]))
        except Exception:
            pass

    if hasattr(ids, "tolist"):
        try:
            return _count_tokens_from_input_ids(ids.tolist())
        except Exception:
            pass

    if isinstance(ids, (list, tuple)):
        if not ids:
            return 0

        first = ids[0]
        if isinstance(first, (list, tuple)):
            return max(0, len(first))

        nested = _count_tokens_from_input_ids(first)
        if nested > 0:
            return nested
        return max(0, len(ids))

    try:
        return max(0, len(ids))
    except Exception:
        return 0


def _estimate_prompt_token_count(pipeline, text: str) -> int:
    text = _normalize_prompt_text_for_count(text)
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text or "",
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
            token_count = _count_tokens_from_input_ids(ids)
            if token_count > 0:
                return token_count
        except Exception:
            pass
    # Fallback heuristic if tokenizer is not available.
    words = len((text or "").strip().split())
    return max(1, int(words * 1.6) + 4)


def _compute_auto_max_sequence_length(
    pipeline,
    prompt: str,
    negative_prompt: str,
    use_cfg: bool,
    hard_cap: int,
) -> int:
    pos_tokens = _estimate_prompt_token_count(pipeline, prompt)
    pos_need = max(64, pos_tokens + 24)

    neg_tokens = 0
    neg_need = 32
    if use_cfg:
        neg_tokens = _estimate_prompt_token_count(pipeline, negative_prompt)
        neg_need = max(32, neg_tokens + 8)

    target = max(pos_need, neg_need if use_cfg else 0)
    chosen = _round_up_to_supported_seq(target, hard_cap)
    print(
        f"[Z-Image POC] Auto max_seq from tokens: pos={pos_tokens}, neg={neg_tokens}, "
        f"use_cfg={use_cfg} -> {chosen} (cap={hard_cap})"
    )
    return chosen


def _universal_zimage_root() -> str:
    return os.path.join(_project_root(), "models", "zimage")


def _is_valid_component_dir(path: str, component_name: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return True
    if component_name == "vae":
        for filename in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
            if os.path.isfile(os.path.join(path, filename)):
                return True
    return False


def _iter_component_dirs(root: str, component_name: str):
    if not root or not os.path.isdir(root):
        return
    direct = os.path.join(root, component_name)
    if os.path.isdir(direct):
        yield direct
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                nested = os.path.join(entry.path, component_name)
                if os.path.isdir(nested):
                    yield nested
    except Exception:
        return


def list_zimage_component_entries(component_name: str, checkpoint_folders: list[str]) -> list[str]:
    if component_name not in ("text_encoder", "vae"):
        return []
    roots = [_universal_zimage_root()] + list(checkpoint_folders or [])
    for folder in list(checkpoint_folders or []):
        try:
            parent = os.path.abspath(os.path.dirname(folder))
            roots.append(os.path.join(parent, "zimage"))
        except Exception:
            pass
    results = []
    seen = set()
    for root in roots:
        for candidate in _iter_component_dirs(root, component_name):
            normalized = os.path.abspath(candidate)
            if normalized in seen:
                continue
            if not _is_valid_component_dir(normalized, component_name):
                continue
            seen.add(normalized)
            results.append(normalized)
    return sorted(results, key=str.casefold)


def _human_component_path(path: str, checkpoint_folders: list[str]) -> str:
    path_abs = os.path.abspath(path)
    roots = [_universal_zimage_root()] + list(checkpoint_folders or [])
    for root in roots:
        if not root:
            continue
        root_abs = os.path.abspath(root)
        prefix = root_abs + os.sep
        if path_abs.startswith(prefix):
            return os.path.relpath(path_abs, root_abs)
    return path_abs


def _component_weight_files(component_dir: str) -> list[str]:
    files = []
    try:
        with os.scandir(component_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name.lower()
                if name.endswith(".safetensors") or name.endswith(".bin") or name.endswith(".pt") or name.endswith(".pth"):
                    files.append(entry.name)
    except Exception:
        return []
    return sorted(files, key=str.casefold)


def _zimage_component_choice_pairs(component_name: str, checkpoint_folders: list[str]) -> list[tuple[str, str]]:
    entries = list_zimage_component_entries(component_name, checkpoint_folders)
    pairs = []
    label_counts = {}

    def _push_choice(raw_label: str, raw_value: str):
        label = raw_label
        if label in label_counts:
            label_counts[label] += 1
            label = f"{label} ({label_counts[label]})"
        else:
            label_counts[label] = 1
        pairs.append((label, raw_value))

    for entry in entries:
        entry = os.path.abspath(entry)
        rel = _human_component_path(entry, checkpoint_folders)
        weight_files = _component_weight_files(entry)
        _push_choice(f"{rel} [default]", entry)
        if component_name == "vae":
            # Single-file VAE overrides are often architecture-incompatible; keep UI on folder/default choices.
            continue
        for filename in weight_files:
            _push_choice(f"{rel} :: {filename}", os.path.abspath(os.path.join(entry, filename)))
    return pairs


def list_zimage_component_choices(component_name: str, checkpoint_folders: list[str]) -> list[str]:
    return [label for label, _ in _zimage_component_choice_pairs(component_name, checkpoint_folders)]


def resolve_zimage_component_path(
    selection: Optional[str],
    component_name: str,
    checkpoint_folders: list[str],
) -> Optional[str]:
    selected = (selection or "").strip()
    if not selected or selected == ZIMAGE_COMPONENT_AUTO:
        return None

    if os.path.isabs(selected):
        if os.path.isfile(selected):
            return os.path.abspath(selected)
        return os.path.abspath(selected) if _is_valid_component_dir(selected, component_name) else None

    for label, path in _zimage_component_choice_pairs(component_name, checkpoint_folders):
        if selected == label:
            return path

    for candidate in list_zimage_component_entries(component_name, checkpoint_folders):
        if candidate == selected:
            return candidate
        if os.path.basename(candidate) == selected:
            return candidate
        parent = os.path.basename(os.path.dirname(candidate))
        if selected == f"{parent}/{component_name}" or selected == f"{parent}\\{component_name}":
            return candidate
    return None


def _forced_zimage_flavor() -> Optional[str]:
    raw = os.environ.get("FOOOCUS_ZIMAGE_FLAVOR", "").strip().lower()
    if raw in ("turbo", "standard"):
        return raw
    if raw:
        _warn_once_env(
            "FOOOCUS_ZIMAGE_FLAVOR",
            f"[Z-Image POC] Ignoring invalid FOOOCUS_ZIMAGE_FLAVOR='{raw}'. Expected 'turbo' or 'standard'.",
        )
    return None


def detect_zimage_flavor(name: str) -> str:
    forced = _forced_zimage_flavor()
    if forced is not None:
        return forced

    # Forge Neo behavior: ZImage checkpoints resolve to Turbo flavor.
    return "turbo"


def _detect_zimage_flavor_from_source(source_kind: str, source_path: str, fallback: str = "turbo") -> str:
    forced = _forced_zimage_flavor()
    if forced is not None:
        return forced

    # Keep function signature stable, but align runtime behavior to Forge Neo:
    # one ZImage class -> Tongyi-MAI/Z-Image-Turbo repo.
    _ = (source_kind, source_path, fallback)
    return "turbo"


def _repo_for_flavor(flavor: str) -> str:
    if flavor != "turbo":
        _warn_once_env(
            "FOOOCUS_ZIMAGE_FLAVOR_NON_TURBO",
            "[Z-Image POC] Non-turbo Z-Image flavor requested; using Turbo repo to match Forge Neo behavior.",
        )
    return "Tongyi-MAI/Z-Image-Turbo"


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

            # Many checkpoints are saved with a transformer prefix
            # (e.g. "model.diffusion_model.") while others are bare keys.
            transformer_prefixes = ("model.diffusion_model.", "diffusion_model.", "transformer.")

            def _strip_prefix(key: str) -> str:
                for prefix in transformer_prefixes:
                    if key.startswith(prefix):
                        return key[len(prefix):]
                return key

            normalized_keys = {_strip_prefix(k) for k in keys}

            if any(k.startswith("text_encoders.qwen3_4b.") for k in keys) or any(
                k.startswith("text_encoders.qwen3_4b.") for k in normalized_keys
            ):
                return True

            cap_weight_key = None
            if "cap_embedder.1.weight" in keys:
                cap_weight_key = "cap_embedder.1.weight"
            else:
                for prefix in transformer_prefixes:
                    candidate = f"{prefix}cap_embedder.1.weight"
                    if candidate in keys:
                        cap_weight_key = candidate
                        break
            if cap_weight_key is not None:
                cap_shape = tuple(f.get_tensor(cap_weight_key).shape)
                # Forge-style detection: Lumina2 backbone with dim=3840 is Z-Image.
                if len(cap_shape) >= 1 and cap_shape[0] == 3840:
                    return True

            has_lumina_backbone = any(k.startswith("layers.0.attention.") for k in normalized_keys)
            has_refiner = any(k.startswith("context_refiner.0.attention.") for k in normalized_keys)
            has_zimage_text = any(k.startswith("text_encoders.") and "qwen3" in k for k in normalized_keys)
            if has_lumina_backbone and (has_refiner or has_zimage_text):
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
    matched, _ = inspect_zimage_checkpoint_detection(name, checkpoint_folders)
    return matched


def inspect_zimage_checkpoint_detection(name: str, checkpoint_folders: list[str]) -> tuple[bool, str]:
    raw_name = str(name or "")
    resolved = _resolve_named_path(raw_name, checkpoint_folders)
    if resolved is None:
        return False, "could not resolve model path from checkpoint folders"

    if os.path.isdir(resolved):
        matched = is_zimage_model_directory(resolved)
        if matched:
            return True, f"resolved directory detected as Z-Image: {resolved}"
        return False, f"resolved directory is not a Z-Image model directory: {resolved}"

    matched = _is_likely_zimage_safetensors(resolved)
    if matched:
        return True, f"resolved safetensors matched Z-Image fingerprint: {resolved}"
    return False, f"resolved safetensors did not match Z-Image fingerprint: {resolved}"


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


def _choose_transformer_mapping(single_file_path: str, state_dict: dict, model_keys: set[str]) -> dict:
    if not state_dict or not model_keys:
        return {"use_forge_mapping": False, "base_score": 0, "converted_score": 0, "cache_hit": False}

    cache_key = _transformer_mapping_cache_key(single_file_path, model_keys)
    cached = _mapping_cache_get(cache_key)
    if cached is not None:
        return {
            "use_forge_mapping": bool(cached.get("use_forge_mapping", False)),
            "base_score": int(cached.get("base_score", 0)),
            "converted_score": int(cached.get("converted_score", 0)),
            "cache_hit": True,
        }

    base_score = _transformer_match_score(state_dict, model_keys)
    converted_keys = []
    for key in state_dict.keys():
        converted_keys.extend(_convert_z_image_transformer_key_to_diffusers(key))
    converted_score = _transformer_match_score_from_keys(converted_keys, model_keys)
    use_forge_mapping = converted_score > base_score

    _mapping_cache_put(
        cache_key,
        {
            "use_forge_mapping": use_forge_mapping,
            "base_score": base_score,
            "converted_score": converted_score,
        },
    )
    return {
        "use_forge_mapping": use_forge_mapping,
        "base_score": base_score,
        "converted_score": converted_score,
        "cache_hit": False,
    }


def _maybe_convert_transformer_checkpoint(single_file_path: str, state_dict: dict, model_keys: set[str]) -> dict:
    if not state_dict or not model_keys:
        return state_dict

    decision = _choose_transformer_mapping(single_file_path, state_dict, model_keys)
    if not decision["use_forge_mapping"]:
        return state_dict

    persisted_sd = _load_persisted_converted_transformer(single_file_path)
    if persisted_sd is not None:
        return persisted_sd

    if decision["cache_hit"]:
        print(
            f"[Z-Image POC] Reusing cached Forge-style Z-Image transformer key mapping "
            f"(score {decision['base_score']}->{decision['converted_score']})."
        )
    else:
        print(
            f"[Z-Image POC] Using Forge-style Z-Image transformer key mapping "
            f"(score {decision['base_score']}->{decision['converted_score']})."
        )
    converted_sd = _convert_z_image_transformer_checkpoint_to_diffusers(state_dict)
    _save_persisted_converted_transformer(single_file_path, converted_sd)
    return converted_sd


def _load_transformer_weights_from_single_file(single_file_path: str, pipeline) -> None:
    parts = _split_single_file_state_dict(single_file_path, include_aux_weights=True)
    transformer_sd = parts["transformer"]
    transformer_component = getattr(pipeline, "transformer", None)
    transformer_model_keys = (
        set(transformer_component.state_dict().keys()) if transformer_component is not None else set()
    )

    if transformer_sd and transformer_model_keys:
        try:
            transformer_sd = _maybe_convert_transformer_checkpoint(
                single_file_path, transformer_sd, transformer_model_keys
            )
        except Exception as e:
            print(f"[Z-Image POC] Z-Image transformer conversion skipped: {e}")

    if not transformer_sd:
        raise RuntimeError(
            "Single-file Z-Image checkpoint does not contain transformer weights in expected format."
        )

    _load_component_override_from_file(
        pipeline,
        "transformer",
        single_file_path,
        state_dict_override=transformer_sd,
        source_label=f"{os.path.basename(single_file_path)}::transformer",
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
    quant_side_suffixes = (
        ".comfy_quant",
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
        ".scale_input",
        ".scale_weight",
    )

    def _map_primary(src_key: str) -> Optional[str]:
        if src_key in model_keys:
            return src_key
        for pref in strip_prefixes[1:]:
            if src_key.startswith(pref):
                cand = src_key[len(pref):]
                if cand in model_keys:
                    return cand
        return None

    remapped = {}
    matched = 0
    for src_key, value in state_dict.items():
        dst_key = _map_primary(src_key)
        if dst_key is None:
            for suffix in quant_side_suffixes:
                if not src_key.endswith(suffix):
                    continue
                base_key = src_key[: -len(suffix)]
                mapped_weight = _map_primary(f"{base_key}.weight")
                if mapped_weight is None or not mapped_weight.endswith(".weight"):
                    continue
                dst_key = mapped_weight[: -len(".weight")] + suffix
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
        try:
            selected_transformer_sd = _maybe_convert_transformer_checkpoint(
                single_file_path, raw_transformer_sd, transformer_model_keys
            )
        except Exception as e:
            print(f"[Z-Image POC] Z-Image transformer conversion skipped: {e}")

    _load_component_override_from_file(
        pipeline,
        "transformer",
        single_file_path,
        state_dict_override=selected_transformer_sd,
        source_label=f"{os.path.basename(single_file_path)}::transformer",
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
            flavor = _detect_zimage_flavor_from_source("directory", resolved, fallback=flavor)
            return "directory", resolved, flavor
        if os.path.isfile(resolved):
            if _is_likely_zimage_safetensors(resolved):
                flavor = _detect_zimage_flavor_from_source("single_file", resolved, fallback=flavor)
                return "single_file", resolved, flavor

    return None, None, flavor


def _pick_device_and_dtype(zimage_allow_fp16: Optional[bool] = None):
    import torch

    dtype_override = _zimage_compute_dtype_mode()
    allow_unsafe_fp16 = _truthy_env("FOOOCUS_ZIMAGE_ALLOW_FP16_UNSAFE", "0")

    def _resolve_dtype(value: str):
        if value in ("bf16", "bfloat16"):
            return torch.bfloat16
        if value in ("fp16", "float16", "half"):
            return torch.float16
        if value in ("fp32", "float32", "full"):
            return torch.float32
        return None

    if torch.cuda.is_available():
        requested = _resolve_dtype(dtype_override)
        if requested is not None:
            # BF16 fallback on GPUs that do not support it.
            if requested == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print("[Z-Image POC] Requested BF16 but CUDA BF16 is unsupported; falling back to FP16.")
                return "cuda", torch.float16
            if requested == torch.float16 and zimage_allow_fp16 is False and not allow_unsafe_fp16:
                print("[Z-Image POC] Requested FP16 but checkpoint appears fp16-unsafe; falling back to FP32.")
                return "cuda", torch.float32
            return "cuda", requested
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        if zimage_allow_fp16 is False and not allow_unsafe_fp16:
            print("[Z-Image POC] Checkpoint appears fp16-unsafe; using FP32.")
            return "cuda", torch.float32
        return "cuda", torch.float16

    requested = _resolve_dtype(dtype_override)
    if requested is not None:
        return "cpu", requested
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


def _choose_memory_mode(device: str, profile: str = "safe") -> tuple[str, float, float, float]:
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

    forced_mode = _zimage_forced_memory_mode()
    if forced_mode is not None:
        return forced_mode, total_vram_gb, free_vram_gb, pressure

    # Profile-based policy:
    # safe: maximize stability under low/medium VRAM
    # balanced: faster default on 10-16GB while keeping fallback room
    # speed: prefer GPU residency/perf, accept higher OOM risk
    if profile == "speed":
        if total_vram_gb > 0 and total_vram_gb <= 8.0:
            return "sequential_offload", total_vram_gb, free_vram_gb, pressure
        if total_vram_gb > 0 and total_vram_gb <= 12.0:
            return "model_offload", total_vram_gb, free_vram_gb, pressure
        if pressure < 0.10:
            return "sequential_offload", total_vram_gb, free_vram_gb, pressure
        if pressure < 0.25:
            return "model_offload", total_vram_gb, free_vram_gb, pressure
        return "full_gpu", total_vram_gb, free_vram_gb, pressure

    if profile == "balanced":
        # 11-12GB cards are still borderline for Z-Image Turbo; default to stricter offload.
        if total_vram_gb > 0 and total_vram_gb <= 12.0:
            return "sequential_offload", total_vram_gb, free_vram_gb, pressure
        if total_vram_gb > 0 and total_vram_gb <= 16.0:
            return "model_offload", total_vram_gb, free_vram_gb, pressure
        if pressure < 0.12:
            return "sequential_offload", total_vram_gb, free_vram_gb, pressure
        if pressure < 0.30:
            return "model_offload", total_vram_gb, free_vram_gb, pressure
        return "full_gpu", total_vram_gb, free_vram_gb, pressure

    # safe profile (default)
    if total_vram_gb > 0 and total_vram_gb <= 12.0:
        return "sequential_offload", total_vram_gb, free_vram_gb, pressure
    if total_vram_gb > 0 and total_vram_gb <= 16.0:
        return "model_offload", total_vram_gb, free_vram_gb, pressure
    if pressure < 0.15:
        return "sequential_offload", total_vram_gb, free_vram_gb, pressure
    if pressure < 0.35:
        return "model_offload", total_vram_gb, free_vram_gb, pressure
    return "full_gpu", total_vram_gb, free_vram_gb, pressure


def _memory_mode_rank(mode: str) -> int:
    return {"full_gpu": 0, "model_offload": 1, "sequential_offload": 2}.get(str(mode), 0)


def _stricter_memory_mode(lhs: str, rhs: str) -> str:
    return lhs if _memory_mode_rank(lhs) >= _memory_mode_rank(rhs) else rhs


def _estimate_generation_vram_need_gb(
    width: int,
    height: int,
    max_sequence_length: int,
    use_cfg: bool,
    flavor: str,
) -> float:
    megapixels = max(0.25, (max(64, int(width)) * max(64, int(height))) / 1_000_000.0)
    base = 4.4 if flavor == "turbo" else 5.8
    pixel_cost = megapixels * 2.0
    seq_cost = max(0.0, float(max_sequence_length) / 256.0) * 1.2
    cfg_cost = 0.4 if use_cfg else 0.0
    estimated = base + pixel_cost + seq_cost + cfg_cost
    return estimated * _zimage_vram_estimate_scale()


def _apply_memory_mode(
    pipeline,
    device: str,
    target_mode: str,
    total_vram_gb: float,
    free_vram_gb: float,
    pressure: float,
    profile: str,
    reason: str = "",
    allow_relax: bool = False,
) -> tuple[str, bool]:
    used_offload = False
    current_mode = getattr(pipeline, "_zimage_memory_mode", "unset")
    if (not allow_relax) and current_mode in ("sequential_offload", "model_offload", "full_gpu"):
        target_mode = _stricter_memory_mode(current_mode, target_mode)

    reason_suffix = f", reason={reason}" if reason else ""
    if target_mode == "sequential_offload" and hasattr(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()
        pipeline._zimage_memory_mode = "sequential_offload"
        used_offload = True
        print(
            f"[Z-Image POC] Using sequential CPU offload "
            f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f}, profile={profile}{reason_suffix})."
        )
    elif target_mode in ("model_offload", "sequential_offload") and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        pipeline._zimage_memory_mode = "model_offload"
        used_offload = True
        print(
            f"[Z-Image POC] Using model CPU offload "
            f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f}, profile={profile}{reason_suffix})."
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
            f"(total={total_vram_gb:.2f}GB, free={free_vram_gb:.2f}GB, pressure={pressure:.2f}, profile={profile}{reason_suffix})."
        )

    return ("cuda" if device == "cuda" else "cpu"), used_offload


def _preflight_generation_memory_mode(
    pipeline,
    cache_key: str,
    device: str,
    generator_device: str,
    used_offload: bool,
    profile: str,
    width: int,
    height: int,
    max_sequence_length: int,
    use_cfg: bool,
    flavor: str,
) -> tuple[str, bool]:
    if device != "cuda" or generator_device != "cuda":
        return generator_device, used_offload

    base_mode, total_vram_gb, free_vram_gb, pressure = _choose_memory_mode(device, profile=profile)
    target_mode = base_mode
    forced_mode = _zimage_forced_memory_mode()
    estimated_need_gb = _estimate_generation_vram_need_gb(
        width=width,
        height=height,
        max_sequence_length=max_sequence_length,
        use_cfg=use_cfg,
        flavor=flavor,
    )
    reserve_vram_gb = _zimage_reserved_vram_gb(total_vram_gb=total_vram_gb)
    estimate_scale = _zimage_vram_estimate_scale()
    headroom_gb = {"safe": 1.75, "balanced": 1.35, "speed": 0.95}.get(profile, 1.35)
    usable_free_gb = max(0.0, free_vram_gb - reserve_vram_gb)
    gap_gb = usable_free_gb - estimated_need_gb

    if forced_mode is None:
        if gap_gb < max(0.35, headroom_gb * 0.50):
            target_mode = "sequential_offload"
        elif gap_gb < headroom_gb:
            target_mode = _stricter_memory_mode(target_mode, "model_offload")
        if flavor == "turbo" and target_mode == "model_offload":
            min_gap_for_model_offload = _zimage_model_offload_min_gap_gb()
            if gap_gb < min_gap_for_model_offload:
                target_mode = "sequential_offload"
    else:
        target_mode = forced_mode

    current_mode = getattr(pipeline, "_zimage_memory_mode", "unset")
    should_reapply_mode = _memory_mode_rank(target_mode) > _memory_mode_rank(current_mode)
    allow_relax = False

    # Allow speed profile to recover from a prior OOM-induced sequential offload once
    # we observe a stable run and enough preflight headroom.
    if (
        forced_mode is None
        and profile == "speed"
        and current_mode == "sequential_offload"
        and target_mode == "model_offload"
        and not bool(getattr(pipeline, "_zimage_last_run_had_oom", False))
        and gap_gb >= max(headroom_gb, 0.9)
    ):
        should_reapply_mode = True
        allow_relax = True

    if forced_mode is not None and target_mode != current_mode:
        should_reapply_mode = True
        allow_relax = True

    if should_reapply_mode:
        reason = f"preflight est={estimated_need_gb:.2f}GB gap={gap_gb:.2f}GB"
        if forced_mode is not None:
            reason = f"forced by env FOOOCUS_ZIMAGE_FORCE_MEMORY_MODE={forced_mode}"
        elif allow_relax:
            reason = f"preflight relax est={estimated_need_gb:.2f}GB gap={gap_gb:.2f}GB"
        try:
            generator_device, used_offload = _apply_memory_mode(
                pipeline=pipeline,
                device=device,
                target_mode=target_mode,
                total_vram_gb=total_vram_gb,
                free_vram_gb=free_vram_gb,
                pressure=pressure,
                profile=profile,
                reason=reason,
                allow_relax=allow_relax,
            )
            _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
        except Exception as e:
            print(
                f"[Z-Image POC] Warning: failed to switch memory mode to '{target_mode}' during preflight: {e}. "
                "Continuing with current mode."
            )
            if current_mode in ("sequential_offload", "model_offload", "full_gpu"):
                pipeline._zimage_memory_mode = current_mode
            _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)

    forced_suffix = f", forced={forced_mode}" if forced_mode is not None else ""
    print(
        f"[Z-Image POC] Preflight VRAM budget: est={estimated_need_gb:.2f}GB, "
        f"est_scale={estimate_scale:.2f}, "
        f"free={free_vram_gb:.2f}GB, reserve={reserve_vram_gb:.2f}GB, usable={usable_free_gb:.2f}GB, "
        f"gap={gap_gb:.2f}GB, base={base_mode}, "
        f"active={getattr(pipeline, '_zimage_memory_mode', 'unset')}{forced_suffix}."
    )
    return generator_device, used_offload


def _is_zimage_pipeline(pipeline) -> bool:
    try:
        class_name = str(getattr(pipeline.__class__, "__name__", "")).lower()
        if "zimage" in class_name:
            return True
    except Exception:
        pass
    try:
        cfg = getattr(pipeline, "config", None)
        if isinstance(cfg, dict):
            cfg_name = str(cfg.get("_class_name", "")).lower()
        else:
            cfg_name = str(getattr(cfg, "_class_name", "")).lower()
        if "zimage" in cfg_name:
            return True
    except Exception:
        pass
    return False


def _disable_xformers_for_pipeline(pipeline, reason: str = "") -> bool:
    changed = False
    if _is_zimage_pipeline(pipeline):
        transformer = getattr(pipeline, "transformer", None)
        if transformer is not None:
            try:
                if hasattr(transformer, "reset_attention_backend"):
                    transformer.reset_attention_backend()
                elif hasattr(transformer, "set_attention_backend"):
                    transformer.set_attention_backend("native")
                changed = True
            except Exception:
                pass
    if hasattr(pipeline, "disable_xformers_memory_efficient_attention"):
        try:
            pipeline.disable_xformers_memory_efficient_attention()
            changed = True
        except Exception:
            pass
    pipeline._zimage_xformers_enabled = False
    pipeline._zimage_xformers_strategy = None
    if changed:
        suffix = f" ({reason})" if reason else ""
        print(f"[Z-Image POC] Disabled xFormers attention{suffix}.")
    return changed


def _maybe_enable_xformers(pipeline, profile: str) -> None:
    mode = _zimage_xformers_mode()
    backend_mode = _zimage_attention_backend_mode()
    explicit_backend = backend_mode != "auto"

    if backend_mode == "native":
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = "native"
        return

    if mode == "off" and backend_mode in ("auto", "xformers"):
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        return
    if not explicit_backend and mode == "auto" and profile not in ("balanced", "speed"):
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        return
    if getattr(pipeline, "_zimage_xformers_attempted", False):
        return
    pipeline._zimage_xformers_attempted = True

    if _is_zimage_pipeline(pipeline):
        transformer = getattr(pipeline, "transformer", None)
        if transformer is not None and hasattr(transformer, "set_attention_backend"):
            base_candidates = _zimage_attention_backend_candidates(
                backend_mode,
                allow_xformers=(mode != "off"),
            )
            discovered = _discover_transformer_attention_backends(transformer)
            candidates = _remap_attention_backend_candidates(base_candidates, discovered)
            if discovered:
                print(f"[Z-Image POC] Attention backend capabilities detected: {discovered}")
                if candidates != base_candidates:
                    print(f"[Z-Image POC] Attention backend alias remap: {base_candidates} -> {candidates}")
            print(
                f"[Z-Image POC] Attention backend probe start: mode={mode}, backend={backend_mode}, "
                f"candidates={candidates}"
            )
            if any(_is_flash_attention_backend(c) for c in candidates):
                print("[Z-Image POC] Flash attention initiation: probing flash-compatible backends.")
            last_error = None
            remapped_from_error = False
            i = 0
            while i < len(candidates):
                candidate = candidates[i]
                if _is_flash_attention_backend(candidate):
                    print(f"[Z-Image POC] Trying flash attention backend '{candidate}'...")
                try:
                    transformer.set_attention_backend(candidate)
                    pipeline._zimage_xformers_enabled = candidate != "native"
                    pipeline._zimage_xformers_strategy = f"dispatch_backend:{candidate}"
                    if candidate == "native":
                        print(
                            f"[Z-Image POC] Using native attention backend for Z-Image "
                            f"(mode={mode}, backend={backend_mode})."
                        )
                    else:
                        print(
                            f"[Z-Image POC] Enabled attention backend '{candidate}' for Z-Image "
                            f"(mode={mode}, backend={backend_mode})."
                        )
                    return
                except Exception as e:
                    last_error = e
                    if _is_flash_attention_backend(candidate):
                        print(f"[Z-Image POC] Flash attention backend '{candidate}' unavailable: {e}")
                    discovered_from_error = _parse_backend_names_from_error(str(e))
                    if discovered_from_error and not remapped_from_error:
                        remapped_from_error = True
                        remapped = _remap_attention_backend_candidates(base_candidates, discovered_from_error)
                        if remapped != candidates:
                            print(
                                f"[Z-Image POC] Attention backend alias remap from runtime error: "
                                f"{base_candidates} -> {remapped}"
                            )
                            candidates = remapped
                            i = 0
                            continue
                i += 1

            pipeline._zimage_xformers_enabled = False
            pipeline._zimage_xformers_strategy = None
            if mode == "on" or explicit_backend:
                print(
                    f"[Z-Image POC] Failed to enable requested attention backend "
                    f"(backend={backend_mode}, mode={mode}): {last_error}"
                )
            return

        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        if mode == "on" or explicit_backend:
            print(
                "[Z-Image POC] Accelerated attention requested but this Z-Image backend lacks "
                "transformer.set_attention_backend(); using native attention."
            )
        return

    if not hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        if mode == "on":
            print("[Z-Image POC] xFormers requested but pipeline does not expose xFormers attention API.")
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        return

    if explicit_backend and backend_mode not in ("xformers",):
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        print(
            f"[Z-Image POC] Attention backend '{backend_mode}' is only supported on Z-Image dispatcher backends; "
            "using native attention."
        )
        return

    try:
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline._zimage_xformers_enabled = True
        pipeline._zimage_xformers_strategy = "processor_swap"
        print(f"[Z-Image POC] Enabled xFormers memory efficient attention (mode={mode}).")
    except Exception as e:
        pipeline._zimage_xformers_enabled = False
        pipeline._zimage_xformers_strategy = None
        if mode == "on":
            print(f"[Z-Image POC] Failed to enable xFormers attention: {e}")


def _should_cleanup_cuda_cache(profile: str, had_oom: bool, pipeline) -> bool:
    if had_oom:
        return True
    if profile == "safe":
        return True

    mode = getattr(pipeline, "_zimage_memory_mode", "unset")
    free_gb, _ = _cuda_mem_info_gb()
    low_free = free_gb > 0 and free_gb < 0.9

    if profile == "balanced":
        # Keep throughput high for offload modes, but still clean up under pressure.
        if mode == "full_gpu" or low_free:
            return True
        return False

    # speed profile: only clean if VRAM is very tight.
    return free_gb > 0 and free_gb < 0.5


def _maybe_preemptive_cuda_cleanup_before_generation(pipeline, profile: str) -> None:
    if not _zimage_preemptive_cuda_cleanup_enabled():
        return

    mode = str(getattr(pipeline, "_zimage_memory_mode", "unset"))
    if mode not in ("model_offload", "sequential_offload", "full_gpu"):
        return

    # Keep speed profile light unless explicitly requested.
    aggressive = _zimage_preemptive_cuda_cleanup_aggressive()
    if profile != "safe" and not aggressive:
        aggressive = False

    free_before, total_gb = _cuda_mem_info_gb()
    try:
        if hasattr(pipeline, "maybe_free_model_hooks"):
            pipeline.maybe_free_model_hooks()
    except Exception:
        pass
    _cleanup_memory(cuda=True, aggressive=aggressive)
    free_after, _ = _cuda_mem_info_gb()

    if free_before > 0 and free_after > 0:
        print(
            f"[Z-Image POC] Pre-run CUDA cleanup: free={free_before:.2f}GB->{free_after:.2f}GB "
            f"(total={total_gb:.2f}GB, mode={mode}, aggressive={aggressive})."
        )


def _prepare_pipeline_memory_mode(pipeline, device: str) -> tuple[str, bool]:
    """
    Returns (generator_device, used_offload_mode).
    generator_device is used to seed torch.Generator.
    """
    import torch

    used_offload = False
    profile = _zimage_perf_profile()
    forced_mode = _zimage_forced_memory_mode()
    pipeline._zimage_perf_profile = profile

    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        _maybe_enable_xformers(pipeline, profile)

        if profile == "safe":
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
        elif profile == "balanced":
            if hasattr(pipeline, "enable_attention_slicing"):
                try:
                    pipeline.enable_attention_slicing("auto")
                except Exception:
                    pipeline.enable_attention_slicing("max")
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
        else:
            # speed profile: avoid forcing slicing/tiling if possible
            if hasattr(pipeline, "disable_attention_slicing"):
                try:
                    pipeline.disable_attention_slicing()
                except Exception:
                    pass

        target_mode, total_vram_gb, free_vram_gb, pressure = _choose_memory_mode(device, profile=profile)
        reason = ""
        if forced_mode is not None:
            reason = f"forced by env FOOOCUS_ZIMAGE_FORCE_MEMORY_MODE={forced_mode}"
        _, used_offload = _apply_memory_mode(
            pipeline=pipeline,
            device=device,
            target_mode=target_mode,
            total_vram_gb=total_vram_gb,
            free_vram_gb=free_vram_gb,
            pressure=pressure,
            profile=profile,
            reason=reason,
            allow_relax=(forced_mode is not None),
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


def _load_component_override(pipeline, component_name: str, component_path: str, dtype) -> None:
    component_path = os.path.abspath(component_path)
    component = getattr(pipeline, component_name, None)
    if component is None:
        print(f"[Z-Image POC] Pipeline has no '{component_name}' component; ignoring override.")
        return
    if os.path.isfile(component_path):
        _load_component_override_from_file(pipeline, component_name, component_path)
        return

    cls = component.__class__
    kwargs = {
        "pretrained_model_name_or_path": component_path,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if component_name == "text_encoder":
        kwargs["trust_remote_code"] = True

    model = _call_with_dtype_compat(
        cls.from_pretrained,
        dtype,
        kwargs,
        f"{component_name}.override.from_pretrained",
    )
    if hasattr(model, "to"):
        model = model.to(dtype=dtype)

    if hasattr(pipeline, "register_modules"):
        pipeline.register_modules(**{component_name: model})
    else:
        setattr(pipeline, component_name, model)
    print(f"[Z-Image POC] Using override {component_name}: {component_path}")


def _load_component_override_from_file(
    pipeline,
    component_name: str,
    component_file: str,
    state_dict_override: Optional[dict] = None,
    source_label: Optional[str] = None,
) -> None:
    import torch

    component = getattr(pipeline, component_name, None)
    if component is None:
        print(f"[Z-Image POC] Pipeline has no '{component_name}' component; ignoring file override.")
        return

    file_path = os.path.abspath(component_file)
    if state_dict_override is not None:
        state_dict = dict(state_dict_override)
    else:
        state_dict = None
        if file_path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as safetensors_load_file
            state_dict = safetensors_load_file(file_path, device="cpu")
        else:
            raw = torch.load(file_path, map_location="cpu")
            if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
                state_dict = raw["state_dict"]
            elif isinstance(raw, dict):
                state_dict = raw
            else:
                raise RuntimeError(f"Unsupported override weights format: {file_path}")

    quant_side_suffixes = (
        ".comfy_quant",
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
        ".scale_input",
        ".scale_weight",
    )

    def _decode_fp4_e2m1_packed_u8(packed: torch.Tensor) -> torch.Tensor:
        # Comfy packs two fp4 values per byte as:
        # packed = (fp4_even << 4) | fp4_odd
        if packed.dtype != torch.uint8:
            packed = packed.to(torch.uint8)
        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
        unpacked[:, 0::2] = hi
        unpacked[:, 1::2] = lo

        # fp4 e2m1 decode table mirrored from Comfy float quantizer behavior.
        table = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ],
            dtype=torch.float32,
            device=packed.device,
        )
        return table[unpacked.long()]

    def _from_blocked_scales(blocked: torch.Tensor) -> torch.Tensor:
        # Inverse of Comfy's to_blocked(...) layout used for NVFP4 block scales.
        if blocked.ndim != 2:
            return blocked
        rows, cols = blocked.shape
        if rows % 128 != 0 or cols % 4 != 0:
            return blocked
        n_row_blocks = rows // 128
        n_col_blocks = cols // 4
        e = blocked.reshape(n_row_blocks * n_col_blocks, 32, 16)
        d = e.reshape(n_row_blocks * n_col_blocks, 32, 4, 4)
        c = d.transpose(1, 2)
        b = c.reshape(n_row_blocks, n_col_blocks, 128, 4)
        a = b.permute(0, 2, 1, 3)
        return a.reshape(rows, cols)

    def _decode_comfy_quant_entry(raw: torch.Tensor) -> Optional[dict]:
        try:
            return json.loads(raw.detach().cpu().numpy().tobytes().decode("utf-8"))
        except Exception:
            return None

    class _ComfyRuntimeQuantLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True, compute_dtype=torch.bfloat16):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.compute_dtype = compute_dtype
            self.quant_format: Optional[str] = None
            self.full_precision_mm = False

            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=compute_dtype), requires_grad=False)
            else:
                self.bias = None

            # Keep optional tensors as plain attrs to avoid None-valued module
            # params/buffers confusing offload hooks.
            self._dense_weight = None
            self._quant_weight = None
            self._weight_scale = None
            self._weight_scale_2 = None
            self._input_scale = None

            self._cached_weight = None
            self._cached_weight_device: Optional[str] = None
            self._cached_weight_dtype: Optional[str] = None

        @property
        def weight(self) -> torch.Tensor:
            # Compatibility shim: some external/offload paths still probe
            # module.weight.{device,dtype} even for custom Linear-like modules.
            if self._dense_weight is not None:
                return self._dense_weight
            if self._quant_weight is not None:
                return self._quant_weight
            # Keep attribute contract even during transient init states.
            return torch.empty(0, dtype=self.compute_dtype)

        @classmethod
        def from_linear(cls, linear_module):
            weight = getattr(linear_module, "weight", None)
            if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise RuntimeError("Module is not linear-like (missing 2D weight tensor).")

            in_features = int(getattr(linear_module, "in_features", weight.shape[1]))
            out_features = int(getattr(linear_module, "out_features", weight.shape[0]))
            bias = getattr(linear_module, "bias", None)
            compute_dtype = getattr(weight, "dtype", torch.bfloat16)

            layer = cls(
                in_features=in_features,
                out_features=out_features,
                bias=bias is not None,
                compute_dtype=compute_dtype,
            )
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias.detach())
            layer._dense_weight = weight.detach()
            return layer

        def _clear_cache(self):
            self._cached_weight = None
            self._cached_weight_device = None
            self._cached_weight_dtype = None

        def _cache_enabled(self) -> bool:
            return _truthy_env("FOOOCUS_ZIMAGE_COMFY_RUNTIME_CACHE", "0")

        def _set_dense_weight(self, weight: torch.Tensor):
            self.quant_format = None
            self._dense_weight = weight.detach()
            self._quant_weight = None
            self._weight_scale = None
            self._weight_scale_2 = None
            self._input_scale = None
            self.full_precision_mm = False
            self._clear_cache()

        def _set_quant_state(
            self,
            fmt: str,
            weight: torch.Tensor,
            weight_scale: Optional[torch.Tensor],
            weight_scale_2: Optional[torch.Tensor],
            input_scale: Optional[torch.Tensor],
            full_precision_mm: bool,
        ):
            self.quant_format = fmt
            self.full_precision_mm = full_precision_mm
            self._quant_weight = weight.detach()
            self._weight_scale = None if weight_scale is None else weight_scale.detach()
            self._weight_scale_2 = None if weight_scale_2 is None else weight_scale_2.detach()
            self._input_scale = None if input_scale is None else input_scale.detach()
            self._dense_weight = None
            self._clear_cache()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            weight_key = f"{prefix}weight"
            bias_key = f"{prefix}bias"
            comfy_quant_key = f"{prefix}comfy_quant"
            weight_scale_key = f"{prefix}weight_scale"
            weight_scale_2_key = f"{prefix}weight_scale_2"
            input_scale_key = f"{prefix}input_scale"
            scale_input_key = f"{prefix}scale_input"
            scale_weight_key = f"{prefix}scale_weight"

            weight = state_dict.pop(weight_key, None)
            bias = state_dict.pop(bias_key, None)
            comfy_quant_raw = state_dict.pop(comfy_quant_key, None)
            weight_scale = state_dict.pop(weight_scale_key, None)
            weight_scale_2 = state_dict.pop(weight_scale_2_key, None)
            input_scale = state_dict.pop(input_scale_key, None)
            if input_scale is None:
                input_scale = state_dict.pop(scale_input_key, None)
            state_dict.pop(scale_weight_key, None)

            if bias is not None and self.bias is not None:
                self.bias.data = bias.detach().to(device=self.bias.device, dtype=self.bias.dtype)

            layer_conf = _decode_comfy_quant_entry(comfy_quant_raw) if comfy_quant_raw is not None else None

            if layer_conf is None:
                if weight is not None:
                    self._set_dense_weight(weight.to(dtype=self.compute_dtype))
                return

            if weight is None:
                raise RuntimeError(f"Quantized layer at '{prefix}' is missing weight tensor.")

            fmt = str(layer_conf.get("format", "")).lower()
            if fmt not in ("float8_e4m3fn", "float8_e5m2", "nvfp4"):
                raise RuntimeError(f"Unsupported Comfy quant format '{fmt}' for layer '{prefix}'.")

            full_precision_mm = bool(layer_conf.get("full_precision_matrix_mult", False))
            self._set_quant_state(
                fmt=fmt,
                weight=weight,
                weight_scale=weight_scale,
                weight_scale_2=weight_scale_2,
                input_scale=input_scale,
                full_precision_mm=full_precision_mm,
            )

        def _dequantize_weight(self, device, dtype):
            if self.quant_format is None:
                if self._dense_weight is None:
                    raise RuntimeError("Dense weight is unavailable.")
                return self._dense_weight.to(device=device, dtype=dtype)

            if self.quant_format in ("float8_e4m3fn", "float8_e5m2"):
                if self._quant_weight is None:
                    raise RuntimeError("FP8 quantized weight is unavailable.")
                weight = self._quant_weight.to(device=device).float()
                if self._weight_scale is not None:
                    weight = weight * self._weight_scale.to(device=device).float()
                return weight.to(dtype=dtype)

            if self.quant_format == "nvfp4":
                if self._quant_weight is None or self._weight_scale is None or self._weight_scale_2 is None:
                    raise RuntimeError("NVFP4 quantized weight/scales are unavailable.")
                deq_values = _decode_fp4_e2m1_packed_u8(self._quant_weight.to(device=device))
                block_scale = _from_blocked_scales(self._weight_scale.to(device=device).float())
                expanded_scale = block_scale.repeat_interleave(16, dim=1) * self._weight_scale_2.to(device=device).float()
                if deq_values.shape != expanded_scale.shape:
                    raise RuntimeError(
                        f"NVFP4 shape mismatch in runtime linear: values={tuple(deq_values.shape)} "
                        f"scales={tuple(expanded_scale.shape)}"
                    )
                return (deq_values * expanded_scale).to(dtype=dtype)

            raise RuntimeError(f"Unknown quant format '{self.quant_format}'.")

        def _resolve_scalar_scale(self, scale_tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
            if scale_tensor is None:
                return torch.ones((), device=device, dtype=torch.float32)
            try:
                scale = scale_tensor.to(device=device, dtype=torch.float32)
            except Exception:
                return None
            if scale.numel() != 1:
                return None
            return scale.reshape(())

        def _apply_weight_scale(self, output: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
            scale = self._weight_scale
            if scale is None:
                return output
            try:
                scale = scale.to(device=device, dtype=output.dtype)
            except Exception:
                return None
            if scale.ndim == 0:
                return output * scale
            if scale.ndim == 1 and scale.shape[0] == self.out_features:
                return output * scale.view(1, -1)
            if scale.ndim == 2 and scale.shape == (self.out_features, 1):
                return output * scale.view(1, -1)
            return None

        def _try_fp8_scaled_mm_linear(self, x: torch.Tensor):
            if not _truthy_env("FOOOCUS_ZIMAGE_COMFY_RUNTIME_FAST_FP8", "1"):
                return None
            if self.quant_format not in ("float8_e4m3fn", "float8_e5m2"):
                return None
            if self.full_precision_mm:
                return None
            if self._quant_weight is None:
                return None
            if not x.is_cuda:
                return None
            if x.ndim != 2:
                return None
            if x.shape[1] != self.in_features:
                return None

            scaled_mm = getattr(torch, "_scaled_mm", None)
            if scaled_mm is None:
                _warn_once_env(
                    "FOOOCUS_ZIMAGE_COMFY_RUNTIME_FAST_FP8",
                    "[Z-Image POC] torch._scaled_mm is unavailable; falling back to standard FP8 linear path.",
                )
                return None

            # cuBLASLt fp8 path requires K and N to be multiples of 16.
            if (self.in_features % 16) != 0 or (self.out_features % 16) != 0:
                return None

            try:
                weight = self._quant_weight.to(device=x.device)
                if weight.ndim != 2:
                    return None
                if weight.shape != (self.out_features, self.in_features):
                    return None

                fp8_dtype = weight.dtype
                if fp8_dtype not in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                    return None

                # Keep values in fp8 finite range before conversion.
                fp8_max = torch.finfo(fp8_dtype).max
                x_fp8 = torch.clamp(x, min=-fp8_max, max=fp8_max).to(dtype=fp8_dtype).contiguous()

                # Use column-major RHS as required by _scaled_mm/cuBLASLt.
                w_t = weight.t()
                if w_t.stride(0) != 1:
                    w_t = weight.contiguous().t()
                    if w_t.stride(0) != 1:
                        return None

                out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.compute_dtype
                scale_a = self._resolve_scalar_scale(self._input_scale, x.device)
                if scale_a is None:
                    return None

                # _scaled_mm tensor-wise mode accepts only singleton scale_b.
                # For non-singleton weight scales, run matmul with scale_b=1 and apply
                # the per-channel scale on the output (same math as current fallback path).
                scale_b = self._resolve_scalar_scale(self._weight_scale, x.device)
                apply_weight_scale_after = scale_b is None and self._weight_scale is not None
                if scale_b is None:
                    scale_b = torch.ones((), device=x.device, dtype=torch.float32)

                bias = self.bias.to(device=x.device, dtype=out_dtype) if self.bias is not None else None

                try:
                    output = scaled_mm(
                        x_fp8,
                        w_t,
                        out_dtype=out_dtype,
                        bias=bias,
                        scale_a=scale_a,
                        scale_b=scale_b,
                    )
                except TypeError:
                    # Older torch builds may not accept bias argument in _scaled_mm.
                    output = scaled_mm(
                        x_fp8,
                        w_t,
                        out_dtype=out_dtype,
                        scale_a=scale_a,
                        scale_b=scale_b,
                    )
                    if bias is not None:
                        output = output + bias

                if isinstance(output, tuple):
                    output = output[0]

                if apply_weight_scale_after:
                    output = self._apply_weight_scale(output, x.device)
                    if output is None:
                        return None
                _warn_once_env(
                    "FOOOCUS_ZIMAGE_COMFY_RUNTIME_FAST_FP8",
                    "[Z-Image POC] Using torch._scaled_mm fast FP8 path for runtime quantized linear layers.",
                )
                return output
            except Exception:
                return None

        def _try_fp8_direct_linear(self, x: torch.Tensor):
            if self.quant_format not in ("float8_e4m3fn", "float8_e5m2"):
                return None
            if self.full_precision_mm:
                return None
            if self._quant_weight is None:
                return None

            # Fast path: torch._scaled_mm for FP8 when supported.
            output = self._try_fp8_scaled_mm_linear(x)
            if output is not None:
                return output

            try:
                weight = self._quant_weight.to(device=x.device)
                output = torch.nn.functional.linear(x, weight, None)
                # Some torch builds can return FP8 here when either operand is FP8.
                # Keep runtime activations in a compute-friendly float dtype.
                if output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    output = output.to(dtype=x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16)
                output = self._apply_weight_scale(output, x.device)
                if output is None:
                    return None

                if self.bias is not None:
                    output = output + self.bias.to(device=x.device, dtype=output.dtype)
                return output
            except Exception:
                return None

        def _runtime_weight(self, x: torch.Tensor):
            force_lowp = _truthy_env("FOOOCUS_ZIMAGE_COMFY_RUNTIME_LOWP", "1")
            if self.quant_format is not None and force_lowp:
                dtype = self.compute_dtype if self.compute_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            else:
                dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.compute_dtype

            # FP16 + FP8 runtime-quant can become numerically unstable on some stacks.
            # Allow stable accumulation dtype while keeping outer runtime in FP16 mode.
            if self.quant_format is not None and dtype == torch.float16:
                mode = _zimage_fp16_quant_accum_mode()
                if mode == "bf16":
                    if x.is_cuda and torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                elif mode == "fp32":
                    dtype = torch.float32
                elif mode == "auto":
                    if x.is_cuda and torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float32

            device_key = str(x.device)
            dtype_key = str(dtype)
            if self._cache_enabled():
                if (
                    self._cached_weight is not None
                    and self._cached_weight_device == device_key
                    and self._cached_weight_dtype == dtype_key
                ):
                    return self._cached_weight

            weight = self._dequantize_weight(device=x.device, dtype=dtype)
            if self._cache_enabled():
                self._cached_weight = weight
                self._cached_weight_device = device_key
                self._cached_weight_dtype = dtype_key
            return weight

        def forward(self, input: torch.Tensor):
            input_shape = input.shape
            x = input.reshape(-1, input_shape[-1]) if input.ndim > 2 else input
            input_runtime_dtype = x.dtype
            force_lowp = _truthy_env("FOOOCUS_ZIMAGE_COMFY_RUNTIME_LOWP", "1")
            if self.quant_format is not None and force_lowp:
                compute_dtype = self.compute_dtype if self.compute_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            else:
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.compute_dtype

            if self.quant_format is not None and compute_dtype == torch.float16:
                mode = _zimage_fp16_quant_accum_mode()
                if mode == "bf16":
                    if x.is_cuda and torch.cuda.is_bf16_supported():
                        compute_dtype = torch.bfloat16
                elif mode == "fp32":
                    compute_dtype = torch.float32
                elif mode == "auto":
                    if x.is_cuda and torch.cuda.is_bf16_supported():
                        compute_dtype = torch.bfloat16
                    else:
                        compute_dtype = torch.float32

            x = x.to(dtype=compute_dtype)

            output = self._try_fp8_direct_linear(x)
            if output is None:
                weight = self._runtime_weight(x)
                bias = self.bias.to(device=x.device, dtype=compute_dtype) if self.bias is not None else None
                output = torch.nn.functional.linear(x, weight, bias)

            # Keep module interface dtype stable, but never propagate FP8 activations.
            if self.quant_format is not None:
                target_dtype = input_runtime_dtype
                if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    target_dtype = compute_dtype if compute_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
                if output.dtype != target_dtype:
                    output = output.to(dtype=target_dtype)

            if input.ndim > 2:
                output = output.reshape(*input_shape[:-1], self.out_features)
            return output

    def _resolve_module(root_module, module_path: str):
        current = root_module
        for part in module_path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def _set_module(root_module, module_path: str, new_module):
        parts = module_path.split(".")
        current = root_module
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        leaf = parts[-1]
        if leaf.isdigit():
            current[int(leaf)] = new_module
        else:
            setattr(current, leaf, new_module)

    def _is_linear_like_module(module) -> bool:
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return False
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            out_features, in_features = weight.shape[0], weight.shape[1]
        try:
            in_features = int(in_features)
            out_features = int(out_features)
        except Exception:
            return False
        return in_features > 0 and out_features > 0

    def _install_comfy_runtime_quant_modules(component_module, remapped_sd: dict) -> dict:
        bases = sorted(
            {
                key[: -len(".comfy_quant")]
                for key in remapped_sd.keys()
                if key.endswith(".comfy_quant") and f"{key[: -len('.comfy_quant')]}.weight" in remapped_sd
            }
        )
        if not bases:
            return {"layers": 0, "replaced": 0, "skipped": 0, "float8": 0, "nvfp4": 0}

        replaced = 0
        skipped = 0
        float8_layers = 0
        nvfp4_layers = 0
        replaced_bases = set()
        skipped_type_counts = {}
        skipped_unresolved = 0

        def _mark_skipped_type(name: str):
            skipped_type_counts[name] = skipped_type_counts.get(name, 0) + 1

        for base in bases:
            conf = _decode_comfy_quant_entry(remapped_sd.get(f"{base}.comfy_quant"))
            fmt = str(conf.get("format", "")).lower() if isinstance(conf, dict) else ""
            if fmt in ("float8_e4m3fn", "float8_e5m2"):
                float8_layers += 1
            elif fmt == "nvfp4":
                nvfp4_layers += 1

            try:
                target = _resolve_module(component_module, base)
            except Exception:
                skipped += 1
                skipped_unresolved += 1
                continue

            if isinstance(target, _ComfyRuntimeQuantLinear):
                replaced += 1
                replaced_bases.add(base)
                continue
            if not _is_linear_like_module(target):
                skipped += 1
                _mark_skipped_type(type(target).__name__)
                continue
            try:
                replacement = _ComfyRuntimeQuantLinear.from_linear(target)
            except Exception:
                skipped += 1
                _mark_skipped_type(type(target).__name__)
                continue
            _set_module(component_module, base, replacement)
            replaced += 1
            replaced_bases.add(base)

        return {
            "layers": len(bases),
            "replaced": replaced,
            "skipped": skipped,
            "float8": float8_layers,
            "nvfp4": nvfp4_layers,
            "replaced_bases": replaced_bases,
            "skipped_unresolved": skipped_unresolved,
            "skipped_type_counts": skipped_type_counts,
        }

    def _normalize_legacy_scaled_fp8_weights(sd: dict) -> tuple[dict, dict]:
        converted = dict(sd)
        migrated = 0
        quant_bases = set()
        for key in list(converted.keys()):
            if key.endswith(".scale_weight"):
                base = key[: -len(".scale_weight")]
                converted[f"{base}.weight_scale"] = converted.pop(key)
                quant_bases.add(base)
                migrated += 1
                continue
            if key.endswith(".scale_input"):
                base = key[: -len(".scale_input")]
                converted[f"{base}.input_scale"] = converted.pop(key)
                quant_bases.add(base)
                migrated += 1

        if migrated == 0:
            return sd, {"migrated": 0, "created_quant_entries": 0}

        created_quant_entries = 0
        for base in quant_bases:
            weight_key = f"{base}.weight"
            scale_key = f"{base}.weight_scale"
            if weight_key not in converted or scale_key not in converted:
                continue
            if f"{base}.comfy_quant" in converted:
                continue
            weight = converted[weight_key]
            fmt = None
            fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
            fp8_e5m2 = getattr(torch, "float8_e5m2", None)
            if fp8_e4m3 is not None and getattr(weight, "dtype", None) == fp8_e4m3:
                fmt = "float8_e4m3fn"
            elif fp8_e5m2 is not None and getattr(weight, "dtype", None) == fp8_e5m2:
                fmt = "float8_e5m2"
            if fmt is None:
                continue
            payload = torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")), dtype=torch.uint8)
            converted[f"{base}.comfy_quant"] = payload
            created_quant_entries += 1

        converted.pop("scaled_fp8", None)
        return converted, {"migrated": migrated, "created_quant_entries": created_quant_entries}

    def _synthesize_native_fp8_quant_entries(sd: dict, component_module=None) -> tuple[dict, dict]:
        converted = dict(sd)
        created = 0
        fmt_counts = {"float8_e4m3fn": 0, "float8_e5m2": 0}
        skipped_non_linear = 0
        skipped_unresolved = 0
        fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        fp8_e5m2 = getattr(torch, "float8_e5m2", None)
        payload_cache = {}

        for key, value in list(converted.items()):
            if not key.endswith(".weight"):
                continue
            base = key[: -len(".weight")]
            if f"{base}.comfy_quant" in converted:
                continue
            dtype = getattr(value, "dtype", None)
            fmt = None
            if fp8_e4m3 is not None and dtype == fp8_e4m3:
                fmt = "float8_e4m3fn"
            elif fp8_e5m2 is not None and dtype == fp8_e5m2:
                fmt = "float8_e5m2"
            if fmt is None:
                continue

            if component_module is not None:
                try:
                    target = _resolve_module(component_module, base)
                except Exception:
                    skipped_unresolved += 1
                    continue
                if not _is_linear_like_module(target):
                    skipped_non_linear += 1
                    continue

            payload = payload_cache.get(fmt)
            if payload is None:
                payload = torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")), dtype=torch.uint8)
                payload_cache[fmt] = payload
            converted[f"{base}.comfy_quant"] = payload
            created += 1
            fmt_counts[fmt] += 1

        return converted, {
            "created": created,
            "skipped_non_linear": skipped_non_linear,
            "skipped_unresolved": skipped_unresolved,
            **fmt_counts,
        }

    def _dequantize_comfy_mixed_weights(sd: dict) -> tuple[dict, dict]:
        quant_entries = {}
        for key in list(sd.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            base = key[: -len(".comfy_quant")]
            conf = _decode_comfy_quant_entry(sd[key])
            if isinstance(conf, dict):
                quant_entries[base] = conf

        if not quant_entries:
            return sd, {"layers": 0, "float8": 0, "nvfp4": 0}

        converted = dict(sd)
        stats = {"layers": 0, "float8": 0, "nvfp4": 0}
        for base, conf in quant_entries.items():
            fmt = str(conf.get("format", "")).lower()
            weight_key = f"{base}.weight"
            if weight_key not in converted:
                continue

            if fmt in ("float8_e4m3fn", "float8_e5m2"):
                weight = converted[weight_key].float()
                scale = converted.get(f"{base}.weight_scale", None)
                if scale is not None:
                    weight = weight * scale.float()
                converted[weight_key] = weight.to(torch.bfloat16)
                stats["float8"] += 1
                stats["layers"] += 1
            elif fmt == "nvfp4":
                packed = converted[weight_key]
                block_scale = converted.get(f"{base}.weight_scale", None)
                tensor_scale = converted.get(f"{base}.weight_scale_2", None)
                if block_scale is None or tensor_scale is None:
                    raise RuntimeError(
                        f"NVFP4 layer '{base}' is missing weight_scale/weight_scale_2."
                    )
                deq = _decode_fp4_e2m1_packed_u8(packed)
                per_block = _from_blocked_scales(block_scale.float())
                expanded_scale = per_block.repeat_interleave(16, dim=1) * tensor_scale.float()
                if deq.shape != expanded_scale.shape:
                    raise RuntimeError(
                        f"NVFP4 shape mismatch in '{base}': values={tuple(deq.shape)} scales={tuple(expanded_scale.shape)}"
                    )
                converted[weight_key] = (deq * expanded_scale).to(torch.bfloat16)
                stats["nvfp4"] += 1
                stats["layers"] += 1
            else:
                raise RuntimeError(f"Unsupported Comfy quant format '{fmt}' in '{base}'.")

            # Remove quant side tensors after dequantization.
            for suffix in quant_side_suffixes:
                converted.pop(f"{base}{suffix}", None)

        # Remove legacy global scaled-fp8 marker if present.
        converted.pop("scaled_fp8", None)
        return converted, stats

    def _dequantize_selected_comfy_bases(sd: dict, selected_bases: set[str]) -> tuple[dict, dict]:
        if not selected_bases:
            return sd, {"selected": 0, "dequantized": 0, "float8": 0, "nvfp4": 0, "skipped": 0}

        converted = dict(sd)
        stats = {
            "selected": len(selected_bases),
            "dequantized": 0,
            "float8": 0,
            "nvfp4": 0,
            "skipped": 0,
        }

        for base in sorted(selected_bases):
            conf = _decode_comfy_quant_entry(converted.get(f"{base}.comfy_quant"))
            weight_key = f"{base}.weight"
            if not isinstance(conf, dict) or weight_key not in converted:
                stats["skipped"] += 1
                continue

            fmt = str(conf.get("format", "")).lower()
            try:
                if fmt in ("float8_e4m3fn", "float8_e5m2"):
                    weight = converted[weight_key].float()
                    scale = converted.get(f"{base}.weight_scale", None)
                    if scale is not None:
                        weight = weight * scale.float()
                    converted[weight_key] = weight.to(torch.bfloat16)
                    stats["float8"] += 1
                    stats["dequantized"] += 1
                elif fmt == "nvfp4":
                    packed = converted[weight_key]
                    block_scale = converted.get(f"{base}.weight_scale", None)
                    tensor_scale = converted.get(f"{base}.weight_scale_2", None)
                    if block_scale is None or tensor_scale is None:
                        stats["skipped"] += 1
                        continue
                    deq = _decode_fp4_e2m1_packed_u8(packed)
                    per_block = _from_blocked_scales(block_scale.float())
                    expanded_scale = per_block.repeat_interleave(16, dim=1) * tensor_scale.float()
                    if deq.shape != expanded_scale.shape:
                        stats["skipped"] += 1
                        continue
                    converted[weight_key] = (deq * expanded_scale).to(torch.bfloat16)
                    stats["nvfp4"] += 1
                    stats["dequantized"] += 1
                else:
                    stats["skipped"] += 1
                    continue
            except Exception:
                stats["skipped"] += 1
                continue

            # Remove quant-side tensors for layers now running eager dense weights.
            for suffix in quant_side_suffixes:
                converted.pop(f"{base}{suffix}", None)

        return converted, stats

    state_dict, legacy_stats = _normalize_legacy_scaled_fp8_weights(state_dict)
    if legacy_stats.get("migrated", 0) > 0:
        print(
            f"[Z-Image POC] Normalized legacy scaled FP8 keys for {component_name}: "
            f"migrated={legacy_stats['migrated']}, quant_entries={legacy_stats['created_quant_entries']}."
        )

    model_keys = set(component.state_dict().keys())
    probe_source_key_count = len(state_dict)
    direct_match_count = 0
    for key in state_dict.keys():
        if key in model_keys:
            direct_match_count += 1
    print(
        f"[Z-Image POC] {component_name} file override key probe: "
        f"source={probe_source_key_count}, direct_matches={direct_match_count}, model_keys={len(model_keys)}"
    )
    remapped = None
    runtime_quant_enabled = (
        component_name in ("text_encoder", "transformer")
        and _truthy_env("FOOOCUS_ZIMAGE_COMFY_RUNTIME_QUANT", "1")
    )
    runtime_stats = {"layers": 0, "replaced": 0, "skipped": 0, "float8": 0, "nvfp4": 0}
    if runtime_quant_enabled:
        remapped_candidate = _remap_state_dict_to_model_keys(
            state_dict,
            model_keys,
            f"{component_name}-file-override-runtime",
            verbose=True,
        )
        remapped_candidate, synth_stats = _synthesize_native_fp8_quant_entries(
            remapped_candidate, component_module=component
        )
        if synth_stats["created"] > 0:
            print(
                f"[Z-Image POC] Synthesized native FP8 quant entries for {component_name}: "
                f"layers={synth_stats['created']}, fp8_e4m3={synth_stats['float8_e4m3fn']}, "
                f"fp8_e5m2={synth_stats['float8_e5m2']}."
            )
        elif synth_stats.get("skipped_non_linear", 0) > 0:
            print(
                f"[Z-Image POC] Native FP8 synth skipped non-linear layers for {component_name}: "
                f"skipped_non_linear={synth_stats['skipped_non_linear']}."
            )
        runtime_stats = _install_comfy_runtime_quant_modules(component, remapped_candidate)
        runtime_stats["backend"] = "runtime"
        if runtime_stats["layers"] > 0 and runtime_stats["replaced"] > 0:
            replaced_bases = runtime_stats.get("replaced_bases", set())
            quant_bases = {
                key[: -len(".comfy_quant")]
                for key in remapped_candidate.keys()
                if key.endswith(".comfy_quant")
            }
            unmapped_bases = quant_bases - set(replaced_bases)
            remapped_candidate, unmapped_stats = _dequantize_selected_comfy_bases(
                remapped_candidate, unmapped_bases
            )
            runtime_stats["unmapped"] = len(unmapped_bases)
            runtime_stats["unmapped_dequantized"] = unmapped_stats.get("dequantized", 0)
            runtime_stats["unmapped_skipped"] = unmapped_stats.get("skipped", 0)
            remapped = remapped_candidate
            if runtime_stats["replaced"] >= runtime_stats["layers"]:
                print(
                    f"[Z-Image POC] Runtime Comfy quant enabled for {component_name}: "
                    f"layers={runtime_stats['layers']}, fp8={runtime_stats['float8']}, "
                    f"nvfp4={runtime_stats['nvfp4']}, backend={runtime_stats.get('backend', 'runtime')}."
                )
            else:
                skipped_types = runtime_stats.get("skipped_type_counts", {})
                skipped_type_summary = ""
                if skipped_types:
                    top = sorted(skipped_types.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    skipped_type_summary = ", skipped_types=" + ",".join(f"{k}:{v}" for k, v in top)
                print(
                    f"[Z-Image POC] Runtime Comfy quant partially mapped for {component_name} "
                    f"(replaced={runtime_stats['replaced']}/{runtime_stats['layers']}, "
                    f"unmapped_dequantized={runtime_stats.get('unmapped_dequantized', 0)}, "
                    f"skipped_unresolved={runtime_stats.get('skipped_unresolved', 0)}"
                    f"{skipped_type_summary}, "
                    "unmapped layers keep eager load path)."
                )

    if remapped is None:
        state_dict, quant_stats = _dequantize_comfy_mixed_weights(state_dict)
        if quant_stats.get("layers", 0) > 0:
            print(
                f"[Z-Image POC] Dequantized Comfy mixed weights for {component_name}: "
                f"layers={quant_stats['layers']}, fp8={quant_stats['float8']}, nvfp4={quant_stats['nvfp4']}."
            )
        remapped = _remap_state_dict_to_model_keys(
            state_dict,
            model_keys,
            f"{component_name}-file-override",
            verbose=True,
        )

    remapped_match_count = 0
    for key in remapped.keys():
        if key in model_keys:
            remapped_match_count += 1
            continue
        for suffix in quant_side_suffixes:
            if key.endswith(suffix):
                weight_key = f"{key[: -len(suffix)]}.weight"
                if weight_key in model_keys:
                    remapped_match_count += 1
                break
    source_key_count = len(remapped)

    precheck_ratio = remapped_match_count / float(max(source_key_count, 1))
    if component_name in ("vae", "text_encoder"):
        if precheck_ratio < 0.35:
            print(
                f"[Z-Image POC] Skipping incompatible {component_name} override file '{source_label or os.path.basename(file_path)}' "
                f"(precheck remap_match={precheck_ratio:.1%}); using model default {component_name}."
            )
            state_dict.clear()
            remapped.clear()
            return

    missing, unexpected = _apply_component_state_dict(
        component,
        remapped,
        label=f"{component_name} file override ({source_label or os.path.basename(file_path)})",
        missing_limit=None,
        unexpected_limit=None,
    )
    # Guardrail: avoid silent garbage generations when incompatible quantized files are selected.
    if model_keys:
        missing_ratio = len(missing) / float(len(model_keys))
    else:
        missing_ratio = 0.0
    unexpected_ratio = len(unexpected) / float(max(len(remapped), 1))
    remap_ratio = remapped_match_count / float(max(source_key_count, 1))
    if remap_ratio < 0.65 or missing_ratio > 0.35 or unexpected_ratio > 0.35:
        if component_name in ("vae", "text_encoder"):
            print(
                f"[Z-Image POC] Skipping incompatible {component_name} override file '{source_label or os.path.basename(file_path)}' "
                f"(remap_match={remap_ratio:.1%}, missing={len(missing)} ({missing_ratio:.1%}), "
                f"unexpected={len(unexpected)} ({unexpected_ratio:.1%})); using model default {component_name}."
            )
            state_dict.clear()
            remapped.clear()
            return
        raise RuntimeError(
            f"Incompatible {component_name} override file '{source_label or os.path.basename(file_path)}': "
            f"remap_match={remap_ratio:.1%}, missing={len(missing)} ({missing_ratio:.1%}), "
            f"unexpected={len(unexpected)} ({unexpected_ratio:.1%}). "
            "Use a compatible full-precision component or the default component folder."
        )
    print(f"[Z-Image POC] Using override {component_name} file: {source_label or file_path}")
    state_dict.clear()
    remapped.clear()


def _load_pipeline(
    source_kind: str,
    source_path: str,
    flavor: str,
    checkpoint_folders: list[str],
    text_encoder_override: Optional[str] = None,
    vae_override: Optional[str] = None,
):
    cache_key = _pipeline_cache_key(
        source_kind,
        source_path,
        text_encoder_override=text_encoder_override,
        vae_override=vae_override,
    )
    if cache_key in _PIPELINE_CACHE:
        pipeline, generator_device, used_offload = _PIPELINE_CACHE[cache_key]
        if _pipeline_has_meta_tensors(pipeline):
            print("[Z-Image POC] Cached pipeline has meta tensors, rebuilding pipeline.")
            _drop_cache_entry(cache_key)
            return _load_pipeline(
                source_kind,
                source_path,
                flavor,
                checkpoint_folders,
                text_encoder_override=text_encoder_override,
                vae_override=vae_override,
            )
        current_profile = _zimage_perf_profile()
        cached_profile = getattr(pipeline, "_zimage_perf_profile", "safe")
        if current_profile != cached_profile:
            device, _ = _pick_device_and_dtype()
            generator_device, used_offload = _prepare_pipeline_memory_mode(pipeline, device)
            _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
        # Keep cached memory mode for throughput; mode hardening is handled at load/OOM time.
        return _PIPELINE_CACHE[cache_key]

    from diffusers import DiffusionPipeline

    _ensure_zimage_runtime_compatibility()
    prefer_single_file_aux_weights = os.environ.get("FOOOCUS_ZIMAGE_LOAD_AIO_AUX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    zimage_allow_fp16 = _detect_zimage_allow_fp16(source_kind, source_path)
    device, dtype = _pick_device_and_dtype(zimage_allow_fp16=zimage_allow_fp16)

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

    if text_encoder_override is not None:
        _load_component_override(pipeline, "text_encoder", text_encoder_override, dtype)
    if vae_override is not None:
        _load_component_override(pipeline, "vae", vae_override, dtype)

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


def _maybe_prewarm_pipeline(
    pipeline,
    generator_device: str,
    flavor: str,
    prewarm_width: Optional[int] = None,
    prewarm_height: Optional[int] = None,
    prewarm_max_sequence_length: Optional[int] = None,
) -> None:
    if not _zimage_prewarm_enabled():
        return
    if bool(getattr(pipeline, "_zimage_prewarm_done", False)):
        return

    # Mark as attempted to avoid repeated startup penalties if warmup fails.
    pipeline._zimage_prewarm_done = True
    pipeline._zimage_prewarm_error = None

    import torch

    try:
        steps = _zimage_prewarm_steps()
        width, height = _zimage_prewarm_size(
            default_width=max(256, int(prewarm_width)) if prewarm_width is not None else 832,
            default_height=max(256, int(prewarm_height)) if prewarm_height is not None else 1216,
        )
        default_max_seq = 64 if flavor == "turbo" else 128
        if prewarm_max_sequence_length is None:
            max_sequence_length = default_max_seq
        else:
            max_sequence_length = max(32, int(prewarm_max_sequence_length))
        prompt = os.environ.get("FOOOCUS_ZIMAGE_PREWARM_PROMPT", "").strip() or "portrait photo"
        negative_prompt = os.environ.get("FOOOCUS_ZIMAGE_PREWARM_NEGATIVE_PROMPT", "").strip()
        guidance = 1.0
        use_cfg = guidance > 1.0

        if generator_device == "cuda":
            profile = _zimage_perf_profile()
            _maybe_preemptive_cuda_cleanup_before_generation(pipeline, profile=profile)

        started = time.perf_counter()
        print(
            f"[Z-Image POC] Prewarm start: steps={steps}, size={width}x{height}, "
            f"max_seq={max_sequence_length}, device={generator_device}."
        )

        with torch.inference_mode():
            pos, neg = pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt if use_cfg else None,
                do_classifier_free_guidance=use_cfg,
                device=generator_device,
                max_sequence_length=max_sequence_length,
            )
            prompt_embeds = [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in pos]
            negative_prompt_embeds = (
                [x.to(device=generator_device, dtype=pipeline.transformer.dtype) for x in neg] if neg else []
            )
            generator = torch.Generator(device=generator_device).manual_seed(1)
            output = _run_pipeline_call(
                pipeline,
                dict(
                    prompt=None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    num_images_per_prompt=1,
                    cfg_normalization=False,
                    cfg_truncation=1.0,
                    max_sequence_length=max_sequence_length,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                ),
            )
            # Force lazy decode/materialization.
            _ = output.images[0]
            del output
            del prompt_embeds
            del negative_prompt_embeds
            del generator

        elapsed = time.perf_counter() - started
        print(f"[Z-Image POC] Prewarm complete in {elapsed:.2f}s.")
    except Exception as e:
        pipeline._zimage_prewarm_error = str(e)
        print(f"[Z-Image POC] Prewarm failed (ignored): {e}")
    finally:
        try:
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
        except Exception:
            pass
        if generator_device == "cuda":
            _cleanup_memory(cuda=True, aggressive=False)


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
    seeds: Optional[list[int]] = None,
    shift: float = 3.0,
    text_encoder_override: Optional[str] = None,
    vae_override: Optional[str] = None,
    return_images: bool = False,
):
    import torch

    stage_timers = _zimage_stage_timers_enabled()
    total_start = time.perf_counter()
    stage_times: dict[str, float] = {}
    embed_cache_hit = False
    generation_attempts = 0
    error_name = ""

    stage_start = time.perf_counter()
    resolved_text_encoder_override = resolve_zimage_component_path(
        text_encoder_override, "text_encoder", checkpoint_folders
    )
    resolved_vae_override = resolve_zimage_component_path(vae_override, "vae", checkpoint_folders)
    stage_times["resolve_overrides"] = time.perf_counter() - stage_start
    cache_key = _pipeline_cache_key(
        source_kind,
        source_path,
        text_encoder_override=resolved_text_encoder_override,
        vae_override=resolved_vae_override,
    )
    profile = _zimage_perf_profile()
    stage_start = time.perf_counter()
    pipeline, generator_device, used_offload = _load_pipeline(
        source_kind,
        source_path,
        flavor,
        checkpoint_folders,
        text_encoder_override=resolved_text_encoder_override,
        vae_override=resolved_vae_override,
    )
    if generator_device == "cuda" and _should_cleanup_cuda_cache(profile, had_oom=False, pipeline=pipeline):
        _cleanup_memory(cuda=True, aggressive=False)
    stage_times["pipeline_load"] = time.perf_counter() - stage_start

    # Align scheduler shift with Forge-style "Shift" control when available.
    try:
        if hasattr(pipeline, "scheduler") and hasattr(pipeline.scheduler, "config"):
            if hasattr(pipeline.scheduler.config, "shift"):
                pipeline.scheduler.config.shift = float(shift)
            if hasattr(pipeline.scheduler, "shift"):
                pipeline.scheduler.shift = float(shift)
    except Exception:
        pass

    # Keep turbo aligned with Z-Image pipeline defaults instead of a stricter local cap.
    max_sequence_length = 512
    use_cfg = guidance_scale > 1.0
    allow_quality_fallback = _zimage_allow_quality_fallback()

    hard_cap = max_sequence_length
    if flavor == "turbo":
        env_max_seq = os.environ.get("FOOOCUS_ZIMAGE_TURBO_MAX_SEQ", "").strip()
        if env_max_seq:
            try:
                env_cap = max(64, int(env_max_seq))
                hard_cap = min(hard_cap, env_cap)
            except Exception:
                pass
    forced_max_seq = getattr(pipeline, "_zimage_forced_max_sequence_length", None)
    if (not allow_quality_fallback) and forced_max_seq is not None:
        forced_max_seq = None
        pipeline._zimage_forced_max_sequence_length = None
    if forced_max_seq is not None:
        hard_cap = min(hard_cap, int(forced_max_seq))

    stage_start = time.perf_counter()
    max_sequence_length = hard_cap
    if flavor == "turbo" and _truthy_env("FOOOCUS_ZIMAGE_DYNAMIC_MAX_SEQ", "1"):
        max_sequence_length = _compute_auto_max_sequence_length(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_cfg=use_cfg,
            hard_cap=hard_cap,
        )

    if generator_device == "cuda" and allow_quality_fallback:
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

    _maybe_prewarm_pipeline(
        pipeline,
        generator_device=generator_device,
        flavor=flavor,
        prewarm_width=width,
        prewarm_height=height,
        prewarm_max_sequence_length=max_sequence_length,
    )

    if generator_device == "cuda":
        _maybe_preemptive_cuda_cleanup_before_generation(pipeline, profile=profile)

    try:
        generator_device, used_offload = _preflight_generation_memory_mode(
            pipeline=pipeline,
            cache_key=cache_key,
            device="cuda" if generator_device == "cuda" else "cpu",
            generator_device=generator_device,
            used_offload=used_offload,
            profile=profile,
            width=width,
            height=height,
            max_sequence_length=max_sequence_length,
            use_cfg=use_cfg,
            flavor=flavor,
        )
    except Exception:
        # Ensure preflight failures don't leave a poisoned cached pipeline for the next image.
        _PIPELINE_CACHE.pop(cache_key, None)
        _clear_prompt_cache_for_pipeline(cache_key)
        raise
    seed_list = [int(seed)]
    if seeds:
        parsed = []
        for s in seeds:
            try:
                parsed.append(int(s))
            except Exception:
                continue
        if parsed:
            seed_list = parsed

    if len(seed_list) <= 1:
        generator = torch.Generator(device=generator_device).manual_seed(seed_list[0])
    else:
        generator = [torch.Generator(device=generator_device).manual_seed(s) for s in seed_list]
    stage_times["runtime_prep"] = time.perf_counter() - stage_start

    neg_key = negative_prompt if use_cfg else ""
    embed_cache_key = (
        cache_key,
        prompt,
        neg_key,
        int(max_sequence_length),
        bool(use_cfg),
    )

    stage_start = time.perf_counter()
    prompt_embeds = None
    negative_prompt_embeds = None
    cached_embeds = _PROMPT_EMBED_CACHE.get(embed_cache_key, None)
    embed_cache_hit = cached_embeds is not None
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
    stage_times["prompt_encode"] = time.perf_counter() - stage_start

    call_kwargs = dict(
        prompt=None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=max(1, len(seed_list)),
        cfg_normalization=False,
        cfg_truncation=1.0,
        max_sequence_length=max_sequence_length,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    print(
        f"[Z-Image POC] Runtime params: steps={steps}, guidance={guidance_scale}, shift={shift}, "
        f"size={call_kwargs['width']}x{call_kwargs['height']}, max_seq={max_sequence_length}, offload={used_offload}, "
        f"batch={call_kwargs['num_images_per_prompt']}, dtype={getattr(pipeline.transformer, 'dtype', 'n/a')}, profile={profile}"
    )

    output = None
    call_start = time.perf_counter()
    black_retry_used = False
    try:
        retry_caps = []
        retry_sizes = []
        if flavor == "turbo" and allow_quality_fallback:
            current_seq = int(call_kwargs.get("max_sequence_length", 256))
            for candidate in (192, 160, 128, 96, 64, 32):
                if current_seq > candidate:
                    retry_caps.append(candidate)
            current_w = int(call_kwargs.get("width", width))
            current_h = int(call_kwargs.get("height", height))
            for scale in (0.85, 0.75, 0.625):
                next_w = max(384, int((current_w * scale) // 64) * 64)
                next_h = max(384, int((current_h * scale) // 64) * 64)
                if next_w < current_w or next_h < current_h:
                    pair = (next_w, next_h)
                    if pair not in retry_sizes:
                        retry_sizes.append(pair)

        max_attempts = 4 if (flavor == "turbo" and allow_quality_fallback) else (2 if flavor == "turbo" else 3)
        for attempt in range(max_attempts):
            generation_attempts = attempt + 1
            try:
                output = _run_pipeline_call(pipeline, call_kwargs)
                if _zimage_black_image_retry_enabled() and not black_retry_used:
                    try:
                        candidates = list(getattr(output, "images", []) or [])
                    except Exception:
                        candidates = []
                    black_entries = []
                    for idx, candidate in enumerate(candidates):
                        try:
                            is_black, black_info = _is_suspected_black_image(candidate)
                        except Exception:
                            is_black, black_info = False, None
                        if is_black and black_info is not None:
                            black_entries.append((idx, black_info))
                    if black_entries:
                        first_black_idx, first_black_info = black_entries[0]
                        is_batch_black = len(candidates) > 1 and len(black_entries) == len(candidates)
                        if len(candidates) == 1 or is_batch_black:
                            black_retry_used = True
                            strategy = str(getattr(pipeline, "_zimage_xformers_strategy", "unknown"))
                            transformer = getattr(pipeline, "transformer", None)
                            transformer_dtype_obj = getattr(transformer, "dtype", None)
                            transformer_dtype = str(transformer_dtype_obj)
                            strict_fp16 = _zimage_strict_fp16_mode() and transformer_dtype_obj == torch.float16
                            if strict_fp16:
                                print(
                                    f"[Z-Image POC] Suspected black output detected "
                                    f"(index={first_black_idx}, mean={first_black_info['mean']:.2f}, "
                                    f"max={first_black_info['max']:.0f}, std={first_black_info['std']:.2f}, "
                                    f"attn={strategy}, dtype={transformer_dtype}). Strict FP16 mode enabled; no fallback."
                                )
                                raise RuntimeError(
                                    "Suspected black output in strict FP16 mode; refusing automatic fallback."
                                )
                            print(
                                f"[Z-Image POC] Suspected black output detected "
                                f"(index={first_black_idx}, mean={first_black_info['mean']:.2f}, "
                                f"max={first_black_info['max']:.0f}, std={first_black_info['std']:.2f}, "
                                f"attn={strategy}, dtype={transformer_dtype}). Retrying once with safer runtime."
                            )

                            changed = []
                            if transformer is not None and hasattr(transformer, "set_attention_backend"):
                                # Flash can occasionally produce pathological outputs on some builds.
                                if "flash" in strategy:
                                    for candidate_backend in ("xformers", "native"):
                                        try:
                                            transformer.set_attention_backend(candidate_backend)
                                            pipeline._zimage_xformers_strategy = f"dispatch_backend:{candidate_backend}"
                                            changed.append(f"attn={candidate_backend}")
                                            break
                                        except Exception:
                                            continue

                            # If user forced fp16 and model behaves badly, attempt one safer dtype retry.
                            if transformer_dtype_obj == torch.float16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                                quant_dtype_updates = 0
                                for module_name in ("transformer", "text_encoder", "vae"):
                                    module = getattr(pipeline, module_name, None)
                                    if module is not None and hasattr(module, "to"):
                                        try:
                                            module.to(dtype=torch.bfloat16)
                                        except Exception:
                                            pass
                                    quant_dtype_updates += _retune_runtime_quant_modules_dtype(module, torch.bfloat16)
                                new_dtype = torch.bfloat16
                                call_kwargs["prompt_embeds"] = [
                                    x.to(device=generator_device, dtype=new_dtype) for x in call_kwargs.get("prompt_embeds", [])
                                ]
                                call_kwargs["negative_prompt_embeds"] = [
                                    x.to(device=generator_device, dtype=new_dtype) for x in call_kwargs.get("negative_prompt_embeds", [])
                                ]
                                changed.append("dtype=bf16")
                                if quant_dtype_updates:
                                    changed.append(f"runtime_quant_dtype={quant_dtype_updates}")

                            if changed:
                                if generator_device == "cuda":
                                    _cleanup_memory(cuda=True, aggressive=True)
                                if len(seed_list) <= 1:
                                    call_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed_list[0])
                                else:
                                    call_kwargs["generator"] = [
                                        torch.Generator(device=generator_device).manual_seed(s) for s in seed_list
                                    ]
                                original_output = output
                                try:
                                    output = _run_pipeline_call(pipeline, call_kwargs)
                                except Exception as retry_error:
                                    output = original_output
                                    print(
                                        f"[Z-Image POC] Black-image retry failed ({retry_error}); keeping original output."
                                    )
                                try:
                                    retry_candidates = list(getattr(output, "images", []) or [])
                                    retry_black_any = False
                                    retry_info = None
                                    for retry_image in retry_candidates:
                                        retry_black, retry_info = _is_suspected_black_image(retry_image)
                                        if retry_black:
                                            retry_black_any = True
                                            break
                                    if retry_black_any and retry_info is not None:
                                        print(
                                            f"[Z-Image POC] Black-image retry remained near-black "
                                            f"(mean={retry_info['mean']:.2f}, max={retry_info['max']:.0f})."
                                        )
                                    else:
                                        print(
                                            f"[Z-Image POC] Black-image retry recovered output using {', '.join(changed)}."
                                        )
                                except Exception:
                                    pass
                            else:
                                print("[Z-Image POC] No safe retry knobs available; keeping original output.")
                        elif black_entries:
                            # For batches, retry only if every output is black. Mixed batches are preserved.
                            print(
                                f"[Z-Image POC] Batch output has {len(black_entries)}/{len(candidates)} near-black images; "
                                "keeping batch output."
                            )
                break
            except RuntimeError as e:
                msg = str(e).lower()

                xformers_mismatch = (
                    "xformersattnprocessor" in msg
                    or "cross_attention_kwargs" in msg
                    or "expanded size of the tensor" in msg
                    or "freqs_cis" in msg
                )
                if xformers_mismatch:
                    disabled = _disable_xformers_for_pipeline(
                        pipeline, reason="runtime mismatch with Z-Image attention kwargs"
                    )
                    if disabled and attempt < 2:
                        print("[Z-Image POC] Retrying after disabling xFormers for Z-Image compatibility.")
                        _cleanup_memory(cuda=(generator_device == "cuda"), aggressive=True)
                        _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
                        continue

                if "out of memory" not in msg or generator_device != "cuda":
                    raise
                if attempt >= (max_attempts - 1):
                    raise

                print("[Z-Image POC] CUDA OOM detected, retrying with stricter offload mode.")
                _cleanup_memory(cuda=True, aggressive=True)
                pipeline._zimage_last_oom = True

                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()
                    pipeline._zimage_memory_mode = "sequential_offload"
                    used_offload = True
                    _PIPELINE_CACHE[cache_key] = (pipeline, generator_device, used_offload)
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing("max")
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()

                if allow_quality_fallback:
                    lowered = False
                    if retry_caps:
                        next_cap = retry_caps.pop(0)
                        call_kwargs["max_sequence_length"] = next_cap
                        pipeline._zimage_forced_max_sequence_length = next_cap
                        print(
                            f"[Z-Image POC] Retrying with reduced max_sequence_length={next_cap} for lower VRAM usage."
                        )
                        lowered = True
                    if retry_sizes:
                        next_w, next_h = retry_sizes.pop(0)
                        call_kwargs["width"] = next_w
                        call_kwargs["height"] = next_h
                        print(
                            f"[Z-Image POC] Retrying with reduced resolution {next_w}x{next_h} for lower VRAM usage."
                        )
                        lowered = True
                    if not lowered:
                        print("[Z-Image POC] Retrying with same sequence length after memory cleanup.")
                else:
                    print("[Z-Image POC] Retrying with same quality settings after memory cleanup.")
                continue

        if output is None:
            raise RuntimeError("Z-Image generation failed after OOM retries.")
        stage_times["pipeline_call"] = time.perf_counter() - call_start

        stage_start = time.perf_counter()
        images = list(getattr(output, "images", []) or [])
        del output
        stage_times["extract_image"] = time.perf_counter() - stage_start
        if return_images:
            return images
        if not images:
            raise RuntimeError("Z-Image pipeline returned no images.")
        return images[0]
    except Exception as e:
        error_name = type(e).__name__
        if "pipeline_call" not in stage_times:
            stage_times["pipeline_call"] = time.perf_counter() - call_start
        # Prevent poisoned/corrupted cache from breaking next generation request.
        _PIPELINE_CACHE.pop(cache_key, None)
        _clear_prompt_cache_for_pipeline(cache_key)
        raise
    finally:
        cleanup_start = time.perf_counter()
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
        had_oom = bool(getattr(pipeline, "_zimage_last_oom", False))
        pipeline._zimage_last_run_had_oom = had_oom
        if hasattr(pipeline, "_zimage_last_oom"):
            pipeline._zimage_last_oom = False
        if generator_device == "cuda":
            if _should_cleanup_cuda_cache(profile, had_oom=had_oom, pipeline=pipeline):
                _cleanup_memory(cuda=True, aggressive=had_oom)
        else:
            _cleanup_memory(cuda=False, aggressive=had_oom)
        stage_times["cleanup"] = time.perf_counter() - cleanup_start
        if stage_timers:
            total_elapsed = time.perf_counter() - total_start
            status = "ok" if not error_name else f"error={error_name}"
            embed_status = "hit" if embed_cache_hit else "miss"
            print(
                f"[Z-Image POC] Stage timings ({status}, embed_cache={embed_status}, attempts={generation_attempts}): "
                f"resolve={_format_timing_ms(stage_times.get('resolve_overrides'))}, "
                f"load={_format_timing_ms(stage_times.get('pipeline_load'))}, "
                f"prepare={_format_timing_ms(stage_times.get('runtime_prep'))}, "
                f"encode={_format_timing_ms(stage_times.get('prompt_encode'))}, "
                f"infer={_format_timing_ms(stage_times.get('pipeline_call'))}, "
                f"extract={_format_timing_ms(stage_times.get('extract_image'))}, "
                f"cleanup={_format_timing_ms(stage_times.get('cleanup'))}, "
                f"total={_format_timing_ms(total_elapsed)}"
            )
