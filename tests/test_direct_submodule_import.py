from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def test_direct_submodule_proxy_imports_static_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_name = "_services_direct_proxy"
    _clearModules(package_name)

    spec = importlib.util.spec_from_file_location(
        package_name,
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module

    try:
        spec.loader.exec_module(module)
        service_module = importlib.import_module(f"{package_name}.chat.service")
        openai_module = importlib.import_module(f"{package_name}.chat.openai")

        assert module.ChatService is service_module.ChatService
        assert (
            service_module.ChatService.__module__
            == f"{package_name}.src.services.chat.service"
        )

        service = service_module.ChatService.fromProvider(
            "openai",
            api_key="test-key",
        )
        assert isinstance(service.generator, openai_module.OpenaiChat)
    finally:
        _clearModules(package_name)


def _clearModules(package_name: str) -> None:
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
