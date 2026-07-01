from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/plugins")
def list_plugins(request: Request) -> list[dict[str, Any]]:
    runtime = request.app.state.plugins
    plugins = [*runtime.vision_plugins, *runtime.control_plugins]
    return [
        {
            "plugin_id": plugin.plugin_id,
            "kind": plugin.kind,
        }
        for plugin in plugins
    ]
