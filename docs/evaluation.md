# Evaluation

`tsuzuri eval` measures whether information given earlier is visible to the model when it becomes
relevant again. It runs every case in an isolated temporary store, offline, so results are reproducible.

## Case format (JSONL, one case per line)

```json
{"id": "plan-interview",
 "agent": "林栖",
 "turns": ["下周三上午我要去面试，有点紧张"],
 "query": "我下周三要做什么，你还记得吗？",
 "expect_any": ["面试"]}
```

| Field | Meaning |
|---|---|
| `id` | `<category>-<name>`; the prefix groups results |
| `agent` | Agent of the one-on-one turns |
| `turns` | Earlier user messages (each is a full turn: reply, fact card, memory write) |
| `room` | Optional `{"members": [...], "turns": [...], "shared": bool}` group conversation |
| `share` | Optional `{"from": agent, "text": ...}` hearsay passed to the probe agent |
| `probe_agent` | Who is asked (defaults to `agent`) |
| `query` | The probe message |
| `expect_any` | Pass if any of these strings is visible to the model |
| `expect_none` | Fail if any of these is visible (privacy / isolation checks) |
| `owner_expect_any` | Isolation control: the same query asked to `agent` (the owner) must show one of these, so a case cannot pass just because nothing is recalled |

"Visible" means: recalled memory rows plus the system messages the compiler produced for the probe
(fact workspace, recalled memory block, state frame) — **not** the persona prompt, so a keyword planted
in a persona cannot pass a case. Hearsay counts only once it reaches the compiled context.
This is a visibility check, not answer accuracy: no reply is generated or judged, so the numbers are
not comparable to end-to-end QA benchmarks such as LoCoMo or LongMemEval. By default the probe is asked in a **new session**
(`"fresh_session": true`): no raw transcript and no session fact cards, so only long-term memory can supply
the answer. Set `"fresh_session": false` to probe inside the same session or room (room members then also
see the room's own context).

## Bundled set

`src/tsuzuri/data/eval_recall_v1.jsonl` covers preferences, upcoming plans, stable facts, corrections
(the newest value must win), per-agent isolation (one agent must not see what the user told another
privately), shared rooms, speaker attribution in rooms, hearsay and multi-turn threads.

```bash
tsuzuri eval                                  # summary
tsuzuri eval --report report.json             # per-case detail
tsuzuri eval --dataset mine.jsonl --min-pass-rate 0.9   # CI gate
```

The offline run tests the memory and compilation pipeline with rule-based extraction. Results with an
LLM extractor (`llm=`) and dense recall (`embeddings=True`) are typically higher on paraphrased probes;
evaluate your own configuration with your own cases before relying on the numbers.

`tsuzuri eval --embeddings` reports `"mode": "embeddings"` only when the dense model actually loaded;
otherwise it says `lexical (embeddings requested but unavailable)`.
