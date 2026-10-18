from __future__ import annotations

import json

from tsuzuri import Tsuzuri, memory_host, persona_registry


CAST = {
    "agents": ["Mira", "Juno"],
    "collective_scope": "crew",
    "collective_noun": "队伍",
    "aliases": {"Mira": ["Mira", "米拉", "小米"], "Juno": ["Juno", "朱诺"]},
    "user": {"name": "Sam", "pronoun": "TA"},
    "settings": {"persona_prompts": {"Mira": "你是米拉。"}},
}


def test_custom_cast_drives_engine_and_prompts(tmp_path):
    memory = Tsuzuri(tmp_path / "store", personas=CAST)
    assert memory.agents == ("Mira", "Juno")
    assert memory.user_name == "Sam"
    turn = memory.chat("Mira", "我下周要搬家")
    assert turn.messages[0]["content"] == "你是米拉。"
    assert "Sam" in turn.fact_card.get("fact", "") + turn.fact_card.get("near_summary", "")


def test_cast_from_json_file(tmp_path):
    path = tmp_path / "cast.json"
    path.write_text(json.dumps(CAST, ensure_ascii=False), encoding="utf-8")
    memory = Tsuzuri(tmp_path / "store", personas=str(path))
    assert memory.agents == ("Mira", "Juno")


def test_registry_basics():
    persona_registry.configure_personas(CAST, reset=True)
    assert persona_registry.resolve_agent("") == "Mira"
    assert persona_registry.resolve_agent(" Juno ") == "Juno"
    assert persona_registry.collective_scope() == "crew"
    assert persona_registry.collective_noun() == "队伍"
    assert "Juno" in persona_registry.agent_set_view()
    assert persona_registry.agent_alias_view()["Mira"] == ["Mira", "米拉", "小米"]
    pattern = persona_registry.persona_regex(r"{USER}和{AGENTS}")
    assert pattern.search("Sam和Juno")
    persona_registry.configure_personas({"user": {"name": "Lee"}})
    assert pattern.pattern != persona_registry.persona_regex(r"{USER}和{AGENTS}").pattern


def test_settings_fall_back_to_defaults():
    persona_registry.configure_personas(CAST, reset=True)
    assert persona_registry.setting("missing", {"a": 1}) == {"a": 1}
    persona_registry.configure_personas({"settings": {"missing": {"b": 2}}})
    assert persona_registry.setting("missing", {"a": 1}) == {"b": 2}


def test_alias_normalization_uses_registry_aliases():
    from tsuzuri.turn_fact_ledger import _canonicalize_character_aliases

    persona_registry.configure_personas(
        {**CAST, "settings": {"turn_fact_alias_whitelist": ["小米"]}}, reset=True,
    )
    assert _canonicalize_character_aliases("小米今天来过") == "Mira今天来过"


def test_host_hook_is_optional_and_overridable():
    assert memory_host.call("agent_system_prompt", "Mira", default="fallback") == "fallback"
    memory_host.register("agent_system_prompt", lambda c: f"prompt for {c}")
    assert memory_host.call("agent_system_prompt", "Mira") == "prompt for Mira"
    memory_host.register("agent_system_prompt", None)
    assert not memory_host.has("agent_system_prompt")
