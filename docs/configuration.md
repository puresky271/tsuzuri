# Configuration

## The cast (`personas`)

Pass a mapping or a JSON path to `Tsuzuri(personas=...)`, set `PERSONA_CONFIG_PATH`, or call `tsuzuri.configure_personas(...)` at runtime. Missing keys fall back to the built-in demo cast.

| Key | Type | Meaning |
|---|---|---|
| `agents` | list of str | Agents the user talks to, in canonical order |
| `default_agent` | str | Used when an API receives an empty agent name (defaults to the first agent) |
| `collective_scope` | str | Partition key of shared, group-level memory |
| `collective_label` | str | Display name of the group in prompts |
| `collective_noun` | str | Common noun for the group in prompts (e.g. "宿舍", "队伍") |
| `collective_keywords` | list of str | Words that name the group |
| `aliases` | {agent: [str]} | Spellings that count as "this agent is mentioned" |
| `text_variants` | {variant: canonical} | Variant spellings folded onto a canonical name before matching |
| `user.name` | str | The user's display name in prompts and fact cards (`DEVELOPER_DISPLAY_NAME` overrides) |
| `user.pronoun` | str | Third-person pronoun for the user ("他" / "她" / "TA") |
| `user.gender` | str | "男" / "女" / "" — enables a pronoun guard in fact extraction |
| `settings` | object | Per-deployment behaviour tables, below |

### `settings`

All optional. Each replaces a built-in default; tables keyed by agent only affect the agents they name.

| Setting | Used by | Shape |
|---|---|---|
| `persona_prompts` | `Tsuzuri`, `Room` | {agent: system prompt} used when no `persona_prompt` is passed |
| `topic_label_keywords` | topic labelling | {label: [keywords]} consulted before the built-in table |
| `topic_labels_vocab` | TurnFact extraction | extra allowed `topic_labels` |
| `hobby_terms` | topic classification | extra hobby words |
| `profile_topic_groups` | reflection de-duplication | [[synonyms]] treated as one topic |
| `collective_context_keywords` | collective writes | words that make a turn group-level when another member is present |
| `collective_event_keywords` | collective writes | words that mark a group event |
| `collective_activity_keywords` | collective writes | words that classify the event as a group activity |
| `entity_topic_hints` | entity recall | topic labels that justify entity hops |
| `location_tokens` | entity typing | extra substrings that mark a name as a place |
| `place_slot_terms` | workspace recall | place names that count as a location slot |
| `agent_full_names` | entity projection | full names of agents (down-weighted as co-occurrence hubs) |
| `known_person_names` | TurnFact | other people's names that count as explicit third parties |
| `agent_name_guards` | TurnFact | {agent: regex} for agent names that are also common words |
| `turn_fact_alias_whitelist` | TurnFact | nicknames always folded onto the canonical name |
| `gossip_profiles` | hearsay | {agent: {"spread_prob": float, "distortion": float}} |
| `ooc_style_rules` | OOC audit | `max_len`, `default_max_len`, `no_soft`, `no_cold`, `banned_tone` |
| `turn_length_budget_tiers` | execution packet | {agent: {"short"/"normal"/"deep": line}} |
| `turn_length_budget_default` | execution packet | the same tiers for agents without an entry |
| `fact_rewrite_examples` | fact rewrite prompt | few-shot block (`{name}`, `{character}`, `{agent_a}`, `{agent_b}`) |
| `rethink_few_shot`, `rethink_insert_few_shot` | rethink prompt | few-shot blocks (`{USER}`, `{PRON}`) |
| `remote_relationship_note` | TurnFact | an extra rule line (`{user}`, `{char}`), e.g. "they only talk by text" |
| `continuity_canonical_key_example` | continuity extraction | example `canonical_key` in the prompt |
| `third_party_example_name` | TurnFact | name used in the third-party example |

See [`examples/custom_cast.json`](../examples/custom_cast.json).

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TSUZURI_HOME` | `./.tsuzuri` | Data directory (set by `Tsuzuri(data_dir=...)`) |
| `PERSONA_CONFIG_PATH` | — | Cast JSON file |
| `DEVELOPER_DISPLAY_NAME` | — | Overrides `user.name` |
| `WORLD_TIMEZONE` / `WORLD_TIMEZONE_LABEL` | `UTC` / `UTC` | World-clock zone (IANA key) and its label in prompts |
| `DEBUG_MOCK_DATE` | — | Pin the world clock (`YYYY-MM-DD[ HH:MM[:SS]]`) |
| `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` | — | OpenAI-compatible endpoint for `openai_compatible()` and internal LLM steps (consolidation). Ignored while a `Tsuzuri(llm=None)` call runs |
| `EMBEDDING_ENABLED` | set by `Tsuzuri(embeddings=...)` | Dense recall. Outside the engine, an existing Hugging Face model cache also switches it on unless set to `0` |
| `MEM_EVOLVE_MIN_SHADOW_DAYS` | `7` | Days after the first consolidation cycle during which evolution only records its plan (set `0` to apply immediately) |
| `MEM_RETHINK_ALLOW_ARCHIVE` | `0` | `1` enables the management API's (soft) delete and rethink archive actions |
| `MEMORY_HOST_MODULE` | `tsuzuri_host` | Module whose `register_hooks(register)` installs host hooks (`none` disables) |
| `TSUZURI_MEMORY_PROVIDER` | `hybrid_v2` | Recall provider (`hybrid_v2`, `biomimetic`) |
| `MEM_MAINTENANCE_SEEDS_PATH` | `<package>/memory_maintenance_seeds.local.json` | Operator-curated seed memories (never commit) |
| `DEVELOPER_USER_PROFILE_PATH` | — | Hand-maintained user facts for the impression editor |

Settings can also live in `$TSUZURI_HOME/tsuzuri.toml` (top-level keys, same names); environment variables win.

### Feature flags

Most flags are read through `secrets_loader.env_bool` (`1/true/yes/on`); a few older ones (`MEM_RETHINK_ALLOW_ARCHIVE`, the entity flags, `MEM_TURN_TOOLS_*`) only accept `1`, so use `1` everywhere. The most relevant:

| Flag | Default | Effect |
|---|---|---|
| `MEM_CLUSTER_MEMORY_ENABLE` (+ `_CREATE_ENABLE`, `_RECALL_ENABLE`) | off | Event/topic cluster memory |
| `MEM_RETHINK_ENABLED` / `MEM_RETHINK_SHADOW_ONLY` | on / off | Semantic rethink during dreaming (needs `llm=`); set `MEM_RETHINK_SHADOW_ONLY=1` to audit without applying |
| `MEM_CORE_SUMMARY_ENABLE` / `MEM_CORE_SUMMARY_RENDER` | off | Core summary rebuild / rendering |
| `MEM_ENTITY_REBUILD_ENABLE`, `MEM_ENTITY_BOOST_ENABLE` | off | Entity projection and recall boost |
| `MEM_FORGETTING_ENABLED` / `MEM_FORGETTING_APPLY` | off / off | Forgetting scan / apply |
| `MEM_EVIDENCE_MAINTENANCE_APPLY` | off | Evidence pool maintenance writes (dry-run otherwise) |
| `MEM_TURN_TOOLS_ENABLE` / `_WRITE_ENABLE` | off | Explicit memory tool round (host-level: call `memory_tools` yourself; `Tsuzuri.chat()` does not run it) |
| `TURN_FACT_SEMANTIC_ENABLE` | on | Semantic channel in the fact workspace (needs embeddings) |
| `DAILY_CONTINUITY_V2_ENABLED` / `_READ_ENABLED` | on | Cross-day continuity capsules (host-level: `chat_compact` produces them; `Tsuzuri.chat()` neither writes nor reads them) |

Forgetting and evidence maintenance default to dry-run. Semantic rethink applies its edits once an LLM is configured **and** the agent has canon anchors (`refresh_character_canon`); otherwise it stays in shadow. Archive actions additionally need `MEM_RETHINK_ALLOW_ARCHIVE=1`. Every edit is recorded in `memory_rethink_runs` / `memory_rethink_actions` and rewritten rows keep their history. Use shadow mode first if you are evaluating a new model.
