from __future__ import annotations

import threading
from typing import Any

import wrapt
from google.genai import live as live_module
from google.genai import types

_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False
_PENDING_TRANSCRIPTION_CONFIGS: dict[int, dict[str, dict[str, Any]]] = {}


def _sanitize_transcription_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    sanitized = dict(config)
    pending: dict[str, dict[str, Any]] = {}
    for key in ("input_audio_transcription", "output_audio_transcription"):
        value = sanitized.get(key)
        if isinstance(value, dict) and value:
            pending[key] = dict(value)
            # Current SDK validates against an empty AudioTranscriptionConfig model.
            sanitized[key] = {}
    return sanitized, pending


def _patched_async_live_connect(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    config = kwargs.get("config")
    using_args = False
    if config is None and len(args) >= 2:
        config = args[1]
        using_args = True

    if isinstance(config, dict):
        sanitized_config, pending = _sanitize_transcription_config(config)
        config_model = types.LiveConnectConfig(**sanitized_config)
        if pending:
            _PENDING_TRANSCRIPTION_CONFIGS[id(config_model)] = pending
        if using_args:
            args = (args[0], config_model, *args[2:])
        else:
            kwargs = dict(kwargs)
            kwargs["config"] = config_model

    return wrapped(*args, **kwargs)


async def _patched_t_live_connect_config(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> types.LiveConnectConfig:
    config = kwargs.get("config")
    using_args = False
    if config is None and len(args) >= 2:
        config = args[1]
        using_args = True

    pending: dict[str, dict[str, Any]] = {}
    pending_from_model: dict[str, dict[str, Any]] = {}
    if isinstance(config, types.LiveConnectConfig):
        pending_from_model = _PENDING_TRANSCRIPTION_CONFIGS.pop(id(config), {})
    if isinstance(config, dict):
        sanitized_config, pending = _sanitize_transcription_config(config)
        if using_args:
            args = (args[0], sanitized_config, *args[2:])
        else:
            kwargs = dict(kwargs)
            kwargs["config"] = sanitized_config

    parameter_model: types.LiveConnectConfig = await wrapped(*args, **kwargs)
    merged_pending = {}
    merged_pending.update(pending_from_model)
    merged_pending.update(pending)
    if merged_pending:
        _PENDING_TRANSCRIPTION_CONFIGS[id(parameter_model)] = merged_pending
    return parameter_model


def _patched_live_connect_parameters_model_dump(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    dumped: dict[str, Any] = wrapped(*args, **kwargs)
    config_model = getattr(instance, "config", None)
    if config_model is None:
        return dumped
    pending = _PENDING_TRANSCRIPTION_CONFIGS.pop(id(config_model), None)
    if not pending:
        return dumped
    config_dump = dumped.setdefault("config", {})
    for key, value in pending.items():
        config_dump[key] = value
    return dumped


def apply_google_genai_live_patch() -> None:
    global _PATCH_APPLIED
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return
        live_module.AsyncLive.connect = wrapt.FunctionWrapper(
            live_module.AsyncLive.connect, _patched_async_live_connect
        )
        live_module._t_live_connect_config = wrapt.FunctionWrapper(
            live_module._t_live_connect_config, _patched_t_live_connect_config
        )
        types.LiveConnectParameters.model_dump = wrapt.FunctionWrapper(
            types.LiveConnectParameters.model_dump,
            _patched_live_connect_parameters_model_dump,
        )
        _PATCH_APPLIED = True
