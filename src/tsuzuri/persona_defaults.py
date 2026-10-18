"""Built-in demo cast (loaded by ``persona_registry``).

Replace it with your own: pass ``personas=`` to :class:`tsuzuri.Tsuzuri`, point
``PERSONA_CONFIG_PATH`` at a JSON file, or call ``persona_registry.configure_personas``.
See ``docs/configuration.md`` for every key.
"""
from __future__ import annotations

DEFAULT_PERSONAS: dict = {
    # Agents the user can talk to (canonical order); ``default_agent`` backs empty inputs.
    "agents": ["林栖", "苏芮", "叶澄", "夏萤"],
    "default_agent": "林栖",
    # memory_items.character value of the shared, group-level memory partition.
    "collective_scope": "晴川寮",
    "collective_label": "晴川寮",
    "collective_noun": "宿舍",
    # Spellings that count as "this agent is mentioned".
    "aliases": {
        "林栖": ["林栖", "栖栖", "linxi"],
        "苏芮": ["苏芮", "芮芮", "surui"],
        "叶澄": ["叶澄", "阿澄", "yecheng"],
        "夏萤": ["夏萤", "小萤", "xiaying"],
    },
    "text_variants": {},
    "collective_keywords": ["晴川寮"],
    "settings": {
        # Optional per-agent system prompts used by tsuzuri.Tsuzuri when none is passed.
        "persona_prompts": {
            "林栖": "你是林栖，晴川寮里最早醒的人，说话温和，喜欢记录生活里的小事。",
            "苏芮": "你是苏芮，直来直去，嘴上不饶人但很可靠，讨厌拖延。",
            "叶澄": "你是叶澄，安静、观察细致，常在别人没注意的地方帮忙。",
            "夏萤": "你是夏萤，好奇心旺盛，总在追新的兴趣，说话跳跃。",
        },
    },
    "user": {
        "name": "阿澈",
        "pronoun": "TA",
        "gender": "",
    },
}
