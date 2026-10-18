"""Compatibility facades for modules split into implementation packages.

``宿主应用``, ``chat_server``, ``town_sim`` and ``memory_core`` used to be single
files. Their code now lives in ``<name>_impl`` packages and the original module
re-exports every former top-level name. Two behaviours keep old callers and
tests working unchanged:

* Attribute writes on the facade (``patch.object(town_sim, "x", ...)``,
  ``monkeypatch.setattr``, ``宿主应用.FLAG = True``) are propagated to every
  submodule that binds the same name, so code inside the package sees them just
  as it did when everything shared one namespace.
  Names that no submodule binds yet (runtime injection) are written to all of
  them, matching the old single-namespace visibility.
* Module state that the package rebinds at runtime via ``global`` statements is
  not copied into the facade; reads and writes are forwarded to the owning
  submodule so the facade never serves a stale value.

New code should import from the specific submodule instead of the facade.
"""
from __future__ import annotations

import sys
import types


class SplitFacadeModule(types.ModuleType):
    def __getattr__(self, name):
        forwarded = self.__dict__.get("__split_forwarded__") or {}
        owner = forwarded.get(name)
        if owner is not None:
            try:
                return owner.__dict__[name]
            except KeyError:
                pass
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "__file__":
            # Parts resolve facade-relative paths through ``_FACADE_FILE``; keep the
            # old ``patch.object(宿主应用, "__file__", tmp)`` redirection working.
            for sub in self.__dict__.get("__split_submodules__", ()):
                if "_FACADE_FILE" in sub.__dict__:
                    sub.__dict__["_FACADE_FILE"] = value
        elif not (name.startswith("__") and name.endswith("__")):
            forwarded = self.__dict__.get("__split_forwarded__") or {}
            owner = forwarded.get(name)
            if owner is not None:
                setattr(owner, name, value)
                return
            subs = self.__dict__.get("__split_submodules__", ())
            bound = [sub for sub in subs if name in sub.__dict__]
            # A name no submodule binds yet (runtime injection, ``patch(create=True)``)
            # was visible to every function of the old single module, so it goes to all.
            for sub in bound or subs:
                sub.__dict__[name] = value
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if not (name.startswith("__") and name.endswith("__")):
            forwarded = self.__dict__.get("__split_forwarded__") or {}
            owner = forwarded.get(name)
            if owner is not None:
                delattr(owner, name)
                return
            for sub in self.__dict__.get("__split_submodules__", ()):
                sub.__dict__.pop(name, None)
        super().__delattr__(name)

    def __dir__(self):
        names = set(super().__dir__())
        names.update(self.__dict__.get("__split_forwarded__") or {})
        return sorted(names)


def install_facade(module_name: str, *, package: str, submodules, forwarded) -> None:
    """Turn ``sys.modules[module_name]`` into a propagating facade."""
    module = sys.modules[module_name]
    subs = tuple(sys.modules[f"{package}.{name}"] for name in submodules)
    module.__dict__["__split_submodules__"] = subs
    module.__dict__["__split_forwarded__"] = {
        name: sys.modules[f"{package}.{owner}"] for name, owner in forwarded.items()
    }
    module.__class__ = SplitFacadeModule


def reexport_optional(namespace: dict, package: str, submodule: str, names) -> None:
    """Copy names a submodule binds only on some paths (e.g. loop variables)."""
    sub = sys.modules[f"{package}.{submodule}"]
    for name in names:
        if name in sub.__dict__:
            namespace[name] = sub.__dict__[name]


def facade_source_files(module_name: str) -> list[str]:
    """Source files making up a (possibly split) module: facade first, then parts."""
    module = sys.modules.get(module_name)
    if module is None:
        module = __import__(module_name)
    files = [module.__file__]
    for sub in module.__dict__.get("__split_submodules__", ()):
        files.append(sub.__file__)
    return files
