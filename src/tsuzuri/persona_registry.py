"""Who the memory system remembers: agents, the shared collective scope, and the user.

Every memory row is partitioned by ``character`` (an agent name) or by the
collective scope; prompts refer to the primary user by display name. Those
identities used to be literals scattered through the engine. They now resolve
here, in this order:

1. ``persona_defaults.DEFAULT_PERSONAS`` (the deployment cast),
2. a JSON file named by ``PERSONA_CONFIG_PATH`` (any subset of keys),
3. env ``DEVELOPER_DISPLAY_NAME`` for the user's display name,
4. ``configure_personas(...)`` at runtime (tests, embedding hosts).

Helpers that compile regexes are cached per config generation, so a runtime
``configure_personas`` call takes effect on the next lookup.
"""
from __future__ import annotations

import collections.abc
import copy
import functools
import json
import os
import re
import threading
from typing import Iterable, Mapping

from tsuzuri.persona_defaults import DEFAULT_PERSONAS

_lock = threading.RLock()
_state: dict = {"config": None, "generation": 0}


def _merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _normalize(raw: Mapping) -> dict:
    agents = [str(a).strip() for a in (raw.get("agents") or []) if str(a or "").strip()]
    default_agent = str(raw.get("default_agent", "") or "").strip()
    if agents and default_agent not in agents:
        default_agent = agents[0]
    elif not agents and default_agent:
        agents = [default_agent]
    aliases: dict[str, list[str]] = {}
    raw_aliases = raw.get("aliases") if isinstance(raw.get("aliases"), Mapping) else {}
    for agent in agents:
        spellings = [str(x).strip() for x in (raw_aliases.get(agent) or []) if str(x or "").strip()]
        aliases[agent] = spellings or [agent]
    variants = raw.get("text_variants") if isinstance(raw.get("text_variants"), Mapping) else {}
    user = raw.get("user") if isinstance(raw.get("user"), Mapping) else {}
    return {
        "agents": tuple(agents),
        "default_agent": default_agent,
        "collective_scope": str(raw.get("collective_scope", "") or "").strip() or "collective",
        "collective_noun": str(raw.get("collective_noun", "") or "").strip() or "小组",
        "collective_label": str(raw.get("collective_label", "") or "").strip()
        or str(raw.get("collective_scope", "") or "").strip() or "collective",
        "aliases": {k: tuple(v) for k, v in aliases.items()},
        "text_variants": {str(k): str(v) for k, v in variants.items() if str(k or "")},
        "collective_keywords": tuple(
            str(x).strip() for x in (raw.get("collective_keywords") or []) if str(x or "").strip()
        ),
        "settings": copy.deepcopy(dict(raw.get("settings") or {})) if isinstance(raw.get("settings"), Mapping) else {},
        "user": {
            "name": str(user.get("name", "") or "").strip() or "User",
            "pronoun": str(user.get("pronoun", "") or "").strip() or "TA",
            "gender": str(user.get("gender", "") or "").strip(),
        },
    }


def _load_initial() -> dict:
    raw = copy.deepcopy(DEFAULT_PERSONAS)
    path = str(os.getenv("PERSONA_CONFIG_PATH", "") or "").strip()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                override = json.load(fh)
            if isinstance(override, Mapping):
                raw = _merge(raw, override)
        except (OSError, ValueError):
            pass
    env_name = str(os.getenv("DEVELOPER_DISPLAY_NAME", "") or "").strip()
    if env_name:
        raw.setdefault("user", {})
        raw["user"] = dict(raw["user"], name=env_name)
    return _normalize(raw)


def _config() -> dict:
    with _lock:
        if _state["config"] is None:
            _state["config"] = _load_initial()
        return _state["config"]


def _denormalize(cfg: Mapping) -> dict:
    return {
        "agents": list(cfg["agents"]),
        "default_agent": cfg["default_agent"],
        "collective_scope": cfg["collective_scope"],
        "collective_label": cfg["collective_label"],
        "collective_noun": cfg["collective_noun"],
        "aliases": {k: list(v) for k, v in cfg["aliases"].items()},
        "text_variants": dict(cfg["text_variants"]),
        "collective_keywords": list(cfg["collective_keywords"]),
        "settings": copy.deepcopy(cfg["settings"]),
        "user": dict(cfg["user"]),
    }


def configure_personas(overrides: Mapping | None = None, *, reset: bool = False) -> dict:
    """Merge ``overrides`` into the active cast (``reset`` starts from defaults). Returns a copy."""
    with _lock:
        base = _load_initial() if (reset or _state["config"] is None) else _state["config"]
        raw = _denormalize(base)
        if overrides:
            raw = _merge(raw, overrides)
        _state["config"] = _normalize(raw)
        _state["generation"] += 1
        return copy.deepcopy(_state["config"])


def generation() -> int:
    _config()
    return int(_state["generation"])


def agent_names() -> tuple[str, ...]:
    return _config()["agents"]


def default_agent() -> str:
    return _config()["default_agent"]


def resolve_agent(name: object) -> str:
    """Stripped ``name`` or the default agent when empty (replaces ``or "<agent>"`` fallbacks)."""
    return str(name or "").strip() or default_agent()


def is_agent(name: object) -> bool:
    return str(name or "").strip() in _config()["agents"]


def collective_scope() -> str:
    return _config()["collective_scope"]


def collective_label() -> str:
    """Display name of the collective (group) in prompts; defaults to the scope id."""
    return _config()["collective_label"]


def collective_noun() -> str:
    """Common noun for the collective in prompts (e.g. "小组", "社团", "小组")."""
    return _config()["collective_noun"]


def collective_keywords() -> tuple[str, ...]:
    return _config()["collective_keywords"]


def agent_aliases(name: object = None) -> dict[str, tuple[str, ...]] | tuple[str, ...]:
    aliases = _config()["aliases"]
    if name is None:
        return dict(aliases)
    key = str(name or "").strip()
    return aliases.get(key, (key,) if key else ())


def text_variants() -> dict[str, str]:
    return dict(_config()["text_variants"])


def fold_text_variants(text: str) -> str:
    out = str(text or "")
    for variant, canonical in _config()["text_variants"].items():
        out = out.replace(variant, canonical)
    return out


def user_name() -> str:
    return _config()["user"]["name"]


def user_pronoun() -> str:
    return _config()["user"]["pronoun"]


def user_gender() -> str:
    return _config()["user"]["gender"]


_CN_DIGITS = "零一二三四五六七八九十"


def count_cn(n: int) -> str:
    """Chinese numeral for small counts used in prompts (0-10), digits beyond."""
    n = int(n)
    return _CN_DIGITS[n] if 0 <= n <= 10 else str(n)


def table_with_extras(key: str, base: Mapping) -> dict:
    """``setting(key)`` entries first (deployment vocabulary), then the built-in ``base`` table."""
    extra = setting(key, {}) or {}
    merged = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in extra.items()} if isinstance(extra, Mapping) else {}
    for k, v in base.items():
        merged.setdefault(k, list(v) if isinstance(v, (list, tuple)) else v)
    return merged


class _LiveTable(collections.abc.Mapping):
    """Read-only ``table_with_extras(key, base)`` that tracks runtime config."""

    def __init__(self, key: str, base: Mapping):
        self._key = key
        self._base = dict(base)

    def _table(self) -> dict:
        return table_with_extras(self._key, self._base)

    def __getitem__(self, item):
        return self._table()[item]

    def __iter__(self):
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())


def live_table(key: str, base: Mapping) -> collections.abc.Mapping:
    return _LiveTable(key, base)


class _LiveList(collections.abc.Set):
    """Read-only set view of ``list_with_extras(key, base)``."""

    def __init__(self, key: str, base: Iterable):
        self._key = key
        self._base = list(base)

    def __iter__(self):
        return iter(list_with_extras(self._key, self._base))

    def __contains__(self, item) -> bool:
        return item in list_with_extras(self._key, self._base)

    def __len__(self) -> int:
        return len(list_with_extras(self._key, self._base))


def live_set(key: str, base: Iterable) -> collections.abc.Set:
    return _LiveList(key, base)


def list_with_extras(key: str, base: Iterable) -> list:
    """Built-in ``base`` items followed by ``setting(key)`` items (deduplicated, order kept)."""
    out: list = []
    for item in [*list(base), *list(setting(key, []) or [])]:
        if item not in out:
            out.append(item)
    return out


def setting(key: str, default=None):
    """Per-deployment behaviour table ``settings[key]`` (e.g. per-agent style rules); ``default`` if unset.

    Callers pass their built-in table as ``default``; a config file can replace it wholesale.
    """
    value = _config()["settings"].get(str(key or ""), None)
    return copy.deepcopy(default) if value is None else copy.deepcopy(value)


class _AgentSetView(collections.abc.Set):
    """Live, read-only set of agent names (tracks ``configure_personas``)."""

    def __iter__(self):
        return iter(agent_names())

    def __contains__(self, item) -> bool:
        return item in agent_names()

    def __len__(self) -> int:
        return len(agent_names())

    def __repr__(self) -> str:
        return f"AgentSetView({list(agent_names())!r})"


class _AgentAliasView(collections.abc.Mapping):
    """Live, read-only mapping agent -> list of alias spellings."""

    def __getitem__(self, key):
        return list(_config()["aliases"][key])

    def __iter__(self):
        return iter(_config()["aliases"])

    def __len__(self) -> int:
        return len(_config()["aliases"])

    def __repr__(self) -> str:
        return f"AgentAliasView({dict(self)!r})"


_AGENT_SET_VIEW = _AgentSetView()
_AGENT_ALIAS_VIEW = _AgentAliasView()


def agent_set_view() -> collections.abc.Set:
    """Drop-in for a module-level ``{agent, ...}`` constant that must follow runtime config."""
    return _AGENT_SET_VIEW


def agent_alias_view() -> collections.abc.Mapping:
    """Drop-in for a module-level ``{agent: [alias, ...]}`` constant that must follow runtime config."""
    return _AGENT_ALIAS_VIEW


def _alternation(values: Iterable[str]) -> str:
    items = sorted({v for v in values if v}, key=lambda v: (-len(v), v))
    return "|".join(re.escape(v) for v in items)


def agents_pattern() -> str:
    """Regex alternation (no group) of all agent names, longest first."""
    return _alternation(agent_names())


_SETTING_PLACEHOLDER = re.compile(r"\{SETTING:([A-Za-z0-9_]+)\}")


def _setting_alternation(match: re.Match) -> str:
    terms = [str(t) for t in (setting(match.group(1), []) or []) if str(t or "")]
    return "(?:" + _alternation(terms) + ")" if terms else "(?!x)x"


@functools.lru_cache(maxsize=256)
def _compile_cached(template: str, flags: int, gen: int) -> re.Pattern:
    del gen
    expanded = (
        template.replace("{USER}", re.escape(user_name()))
        .replace("{AGENTS}", "(?:" + agents_pattern() + ")")
        .replace("{COLLECTIVE_LOWER}", re.escape(collective_scope().lower()))
        .replace("{COLLECTIVE_NOUN}", re.escape(collective_noun()))
        .replace("{COLLECTIVE}", re.escape(collective_scope()))
    )
    return re.compile(_SETTING_PLACEHOLDER.sub(_setting_alternation, expanded), flags)


def persona_regex(template: str, flags: int = 0) -> re.Pattern:
    """Compile ``template`` with ``{USER}`` / ``{AGENTS}`` / ``{COLLECTIVE}`` filled from the cast.

    Placeholders are literal tokens (not ``str.format``), so regex braces such as ``{1,8}`` stay intact.
    Also: ``{COLLECTIVE_LOWER}``, ``{COLLECTIVE_NOUN}`` and ``{SETTING:key}`` (alternation of the
    string list ``setting(key)``; never matches when the list is empty).
    """
    return _compile_cached(template, flags, generation())


class LivePattern:
    """A module-level regex constant whose ``{USER}`` / ``{AGENTS}`` / ``{COLLECTIVE}`` follow the cast.

    Quacks like ``re.Pattern`` for the methods the engine uses; compiled lazily and
    re-compiled after ``configure_personas``.
    """

    __slots__ = ("_template", "_flags")

    def __init__(self, template: str, flags: int = 0):
        self._template = template
        self._flags = flags

    @property
    def compiled(self) -> re.Pattern:
        return persona_regex(self._template, self._flags)

    @property
    def pattern(self) -> str:
        return self.compiled.pattern

    @property
    def flags(self) -> int:
        return self.compiled.flags

    def search(self, *args, **kwargs):
        return self.compiled.search(*args, **kwargs)

    def match(self, *args, **kwargs):
        return self.compiled.match(*args, **kwargs)

    def fullmatch(self, *args, **kwargs):
        return self.compiled.fullmatch(*args, **kwargs)

    def sub(self, *args, **kwargs):
        return self.compiled.sub(*args, **kwargs)

    def subn(self, *args, **kwargs):
        return self.compiled.subn(*args, **kwargs)

    def findall(self, *args, **kwargs):
        return self.compiled.findall(*args, **kwargs)

    def finditer(self, *args, **kwargs):
        return self.compiled.finditer(*args, **kwargs)

    def split(self, *args, **kwargs):
        return self.compiled.split(*args, **kwargs)

    def __repr__(self) -> str:
        return f"LivePattern({self._template!r})"


def live_pattern(template: str, flags: int = 0) -> LivePattern:
    return LivePattern(template, flags)


__all__ = [
    "configure_personas", "generation",
    "agent_names", "default_agent", "resolve_agent", "is_agent",
    "collective_scope", "collective_label", "collective_noun", "collective_keywords",
    "agent_aliases", "agent_set_view", "agent_alias_view", "text_variants", "fold_text_variants",
    "user_name", "user_pronoun", "user_gender", "setting", "count_cn", "table_with_extras", "list_with_extras", "live_table", "live_set",
    "agents_pattern", "persona_regex", "live_pattern", "LivePattern",
]
