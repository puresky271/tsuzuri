"""Tests for [QueryExpand] 规则版语句切分 (2026-08-04).

Covers:
1. Pure functions: _split_query_clauses / _extract_query_keywords /
   _split_query_semantics / _expand_query_semantically / _rule_expand_query.
2. Real-DB pool feeding: 变体把老记忆（掉出 recency 窗口）拉进 FTS/LIKE 候选池。
3. Real retrieve-level A/B (embedding 可用时): 非显式 query 下、
   老记忆只有开启 expansion 才被召回——结构性失效的端到端验证。

Project rule: real recall detection is mandatory — mock-only tests do not count.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# 注意：不要在模块级改 EMBEDDING_ENABLED——pytest 会话共享 os.environ，
# 全局改掉会让同会话其他测试文件（guest/cross_char 需要 embedding）静默失效。
# 改为各测试类 setUp 内保存/恢复。
os.environ.setdefault("SEMANTIC_QUERY_EXPAND", "1")

sys.path.insert(0, os.path.dirname(__file__))

from tsuzuri import memory_core as mc
from tsuzuri.memory_core import (
    MemoryCoreStore,
    _expand_query_semantically,
    _extract_query_keywords,
    _rule_expand_query,
    _split_query_clauses,
    _split_query_semantics,
)


@contextmanager
def _env(key: str, value: str):
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _utc_iso(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _make_payload(ew: float = 0.3) -> str:
    return json.dumps({
        "memory_policy": {"memory_features": {"emotional_weight": ew}},
    }, ensure_ascii=False)


def _restore_env(key: str, prev: str | None) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


def _seed_fillers(store: MemoryCoreStore, n: int = 250, char: str = "张三") -> None:
    """Raw-SQL 批量插入最近 fillers（绕开 embedding/FTS 成本，只用于撑满 recency 窗口）。"""
    now = _utc_iso()
    cols = ("item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, "
            "cue_json, lifecycle_json, salience, confidence, novelty, importance, status, source, "
            "evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, "
            "embedding_text, topic_labels_json")
    rows = [(
        f"fill_{i}", char, "episodic", "episodic", "event", "event",
        f"普通的日常记录第{i}条，买面包回家看电视", "{}", "{}", "{}",
        0.5, 0.7, 0.4, 0.5, "active", "test", "", now, now, "", 1, 0, "", "[]",
    ) for i in range(n)]
    conn = store._connect()
    try:
        conn.executemany(f"INSERT INTO memory_items ({cols}) VALUES ({','.join(['?'] * 24)})", rows)
        conn.commit()
    finally:
        conn.close()


def _insert_item(store: MemoryCoreStore, item_id: str, text: str, days_ago: int) -> None:
    ts = _utc_iso(days_ago)
    store._upsert_item({
        "item_id": item_id, "character": "张三", "layer": "episodic", "memory_role": "episodic",
        "engram_type": "event", "item_type": "event", "text": text,
        "payload_json": _make_payload(),
        "salience": 0.6, "confidence": 0.75, "novelty": 0.4, "importance": 0.7,
        "status": "active", "source": "test", "created_at": ts, "updated_at": ts, "access_count": 1,
    })


def _fresh_store(tmp: str) -> MemoryCoreStore:
    db = os.path.join(tmp, "t.db")
    snap = os.path.join(tmp, "s.json")
    with open(snap, "w") as f:
        json.dump({}, f)
    return MemoryCoreStore(db_path=db, snapshot_path=snap)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 纯函数：子句拆分
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitQueryClauses(unittest.TestCase):

    def test_punctuation_split(self):
        self.assertEqual(_split_query_clauses("集训完，赵六先走了"),
                         ["集训完", "赵六先走了"])

    def test_conjunction_split(self):
        # 时间连词拆开；"之后" 拆进后段首、剥掉段首连词
        self.assertEqual(_split_query_clauses("上次集训吵架之后赵六说了什么"),
                         ["上次集训吵架", "赵六说了什么"])

    def test_conjunction_at_start_lead_stripped(self):
        self.assertEqual(_split_query_clauses("然后我们去了活动中心"), ["我们去了活动中心"])

    def test_short_sides_do_not_create_generic_variant(self):
        self.assertEqual(_split_query_clauses("我们之后再说吧"), ["我们之后再说吧"])

    def test_conjunction_idiom_is_not_split(self):
        self.assertEqual(_split_query_clauses("赵六后来居上了吗"), ["赵六后来居上了吗"])

    def test_idiom_guard_does_not_block_other_conjunctions(self):
        self.assertEqual(
            _split_query_clauses("上次集训之后赵六后来居上了吗"),
            ["上次集训", "赵六后来居上了吗"],
        )

    def test_short_leading_conjunction_is_preserved(self):
        self.assertEqual(_split_query_clauses("还有珠吗"), ["还有珠吗"])

    def test_no_over_split_simple_query(self):
        self.assertEqual(_split_query_clauses("晴天小调那首歌你还记得吗"),
                         ["晴天小调那首歌你还记得吗"])

    def test_min_length_filter(self):
        self.assertEqual(_split_query_clauses("，。"), [])

    def test_dedup(self):
        self.assertEqual(_split_query_clauses("集训。集训"), ["集训"])

    def test_max_clauses_cap(self):
        out = _split_query_clauses("a，b，c，d，e，f", max_clauses=3)
        self.assertLessEqual(len(out), 3)

    def test_empty_input(self):
        self.assertEqual(_split_query_clauses(""), [])


# ─────────────────────────────────────────────────────────────────────────────
# 2. 纯函数：关键词抽取
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractQueryKeywords(unittest.TestCase):

    def test_character_name(self):
        self.assertIn("赵六", _extract_query_keywords("赵六最近怎么样", "张三"))

    def test_domain_keyword(self):
        self.assertIn("集训", _extract_query_keywords("集训的时候发生了什么", "张三"))

    def test_jieba_noun_entity(self):
        # 固定字典外的具体实体，靠 jieba 名词层抽取
        self.assertIn("水族馆", _extract_query_keywords("我们之前说好的水族馆还在营业吗", "张三"))

    def test_single_char_alias_not_extracted(self):
        # "李四" 单字别名粒度低（灯笼误命中）→ 不抽
        self.assertNotIn("李四", _extract_query_keywords("灯笼很亮", "张三"))

    def test_stopword_excluded(self):
        self.assertNotIn("发生", _extract_query_keywords("发生了什么", "张三"))
        self.assertNotIn("事情", _extract_query_keywords("那件事情", "张三"))

    def test_developer_name(self):
        self.assertIn("阿澈", _extract_query_keywords("阿澈最近在忙什么", "张三"))

    def test_max_keywords_bound(self):
        out = _extract_query_keywords("赵六和王五集训吵架考试打工水族馆", "张三", max_keywords=3)
        self.assertLessEqual(len(out), 3)

    def test_empty_input(self):
        self.assertEqual(_extract_query_keywords(""), [])

    def test_first_person_fact_query_drops_character_name(self):
        # 2026-09-14 郁金香失败案例：称呼语位置的角色名对"我 X 是什么"零判别力，
        # 裸名变体会灌大角色档案事实、挤掉真正答案。
        out = _extract_query_keywords("张三，你还记得我最喜欢的花是哪一种吗？", "张三")
        self.assertNotIn("张三", out)

    def test_character_topic_query_keeps_character_name(self):
        # 以角色为主题的问句（无第一人称标记）不受影响，名字照常保留
        self.assertIn("张三", _extract_query_keywords("张三的生日是哪一天", "张三"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 纯函数：切分主入口 + 变体
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitQuerySemantics(unittest.TestCase):

    def test_disabled_returns_empty(self):
        with _env("SEMANTIC_QUERY_EXPAND", "0"):
            s = _split_query_semantics("集训吵架", "张三")
            self.assertEqual(s, {"clauses": [], "keywords": [], "variants": []})

    def test_compound_query_decomposed(self):
        s = _split_query_semantics("上次集训吵架之后赵六说了什么", "张三")
        self.assertIn("赵六说了什么", s["clauses"])
        self.assertTrue(s["keywords"])

    def test_variants_original_first_bounded(self):
        exps = _expand_query_semantically("上次集训吵架之后赵六说了什么", "张三")
        self.assertEqual(exps[0], "上次集训吵架之后赵六说了什么")
        self.assertLessEqual(len(exps), 3)
        self.assertEqual(len(set(exps)), len(exps))  # 去重


class TestExpandQuerySemanticallyGate(unittest.TestCase):

    def test_disabled_returns_single(self):
        with _env("SEMANTIC_QUERY_EXPAND", "0"):
            self.assertEqual(_expand_query_semantically("集训吵架", "张三"), ["集训吵架"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. 纯函数：同义替换
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleExpandQuery(unittest.TestCase):

    def test_synonym_replacement(self):
        vs = _rule_expand_query("还记得上次看演唱会吗", "张三")
        self.assertIn("还记得上次看live吗", vs)
        self.assertIn("还记得上次看演出吗", vs)  # 2026-08-04: 不再只取 expansions[:1]

    def test_session_synonym(self):
        self.assertTrue(any("练习" in v for v in _rule_expand_query("今天集训了吗", "张三")))

    def test_no_matches_returns_empty(self):
        self.assertEqual(_rule_expand_query("完全无关的一句话", "张三"), [])

    def test_short_query_returns_empty(self):
        self.assertEqual(_rule_expand_query("集训", "张三"), [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. 真实 DB：变体回喂 FTS/LIKE 候选池（结构性修复，embedding 无关、确定性）
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryPoolFeeding(unittest.TestCase):
    """250 条最近 fillers 把 40 天前的 target 挤出 recency 窗口（limit 220），
    只有 extra_search_texts（同义变体）能通过 jieba LIKE 把它拉回候选池。"""

    def setUp(self):
        self._prev_emb = os.environ.get("EMBEDDING_ENABLED")
        os.environ["EMBEDDING_ENABLED"] = "false"  # 池测试只查 SQL 候选池、不需要 embedding
        self._tmp = tempfile.mkdtemp()
        self._store = _fresh_store(self._tmp)
        _seed_fillers(self._store, n=250)
        _insert_item(self._store, "old_target", "王五说周末要去看演出，赵六说要一起去", days_ago=40)
        _insert_item(self._store, "old_deferral_noise", "我们上次集训后去吃了拉面", days_ago=40)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        _restore_env("EMBEDDING_ENABLED", self._prev_emb)

    def test_target_out_of_pool_without_extra(self):
        rows = self._store._query_items("张三", limit=220, query_text="演唱会")
        self.assertNotIn("old_target", [r["item_id"] for r in rows])

    def test_synonym_extra_pulls_target_into_pool(self):
        rows = self._store._query_items("张三", limit=220, query_text="演唱会",
                                        extra_search_texts=("演出", "live"))
        self.assertIn("old_target", [r["item_id"] for r in rows])

    def test_retrieve_builds_expected_extra(self):
        # 检索路径实际喂给 _query_items 的扩展词应包含同义变体（含"演出"）
        with _env("SEMANTIC_QUERY_EXPAND", "1"):
            sem = _split_query_semantics("演唱会安排", "张三")
            syns = _rule_expand_query("演唱会安排", "张三")
        self.assertTrue(any("演出" in s for s in syns))
        self.assertTrue(sem["clauses"] or sem["keywords"] or syns)

    def test_deferral_phrase_does_not_surface_generic_old_memory(self):
        with _env("SEMANTIC_QUERY_EXPAND", "1"):
            result = self._store.retrieve_memory_context(
                "张三", "我们之后再说吧", query_bundle={}, max_items=5,
                touch_selected=False, log_retrieval=False,
            )
        self.assertNotIn("old_deferral_noise", [r.get("item_id") for r in result.get("rows", [])])


class TestCollectivePoolFeeding(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._store = _fresh_store(self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_llm_variant_is_forwarded_to_collective_candidate_pool(self):
        with (_env("SEMANTIC_QUERY_EXPAND", "1"),
              patch.object(mc, "_expand_query_semantically", return_value=["演唱会安排", "周末舞台计划"]),
              patch.object(self._store, "_query_items", return_value=[]) as query_items):
            self._store.retrieve_collective_memory_context("张三", "演唱会安排")

        extra = query_items.call_args.kwargs["extra_search_texts"]
        self.assertIn("周末舞台计划", extra)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 真实 DB + 真实检索（embedding 可用时）：端到端 A/B
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveLevelExpansion(unittest.TestCase):
    """非显式 query 命中 40 天前的记忆：开启 expansion 才召回。
    生产默认 embedding ON；本类只有 embedding 真正可用时才运行。"""

    @classmethod
    def setUpClass(cls):
        cls._prev_emb = os.environ.get("EMBEDDING_ENABLED")
        os.environ["EMBEDDING_ENABLED"] = "true"
        try:
            cls._emb_ok = mc._emb_encode("测试") is not None
        except Exception:
            cls._emb_ok = False
        finally:
            _restore_env("EMBEDDING_ENABLED", cls._prev_emb)

    def setUp(self):
        self._prev_emb = os.environ.get("EMBEDDING_ENABLED")
        os.environ["EMBEDDING_ENABLED"] = "true"
        self._tmp = tempfile.mkdtemp()
        self._store = _fresh_store(self._tmp)
        try:
            # fillers 走 _upsert_item：嵌入模型已 warm、批量预编码、
            # 避免检索时 250 条 on-the-fly backfill 拖慢 A/B 两次调用
            now = _utc_iso()
            for i in range(250):
                self._store._upsert_item({
                    "item_id": f"fill_{i}", "character": "张三", "layer": "episodic",
                    "memory_role": "episodic", "engram_type": "event", "item_type": "event",
                    "text": f"普通的日常记录第{i}条，买面包回家看电视",
                    "payload_json": _make_payload(), "salience": 0.5, "confidence": 0.7,
                    "novelty": 0.4, "importance": 0.5, "status": "active", "source": "test",
                    "created_at": now, "updated_at": now, "access_count": 1,
                })
            _insert_item(self._store, "old_target", "王五说周末要去看演出，赵六说要一起去", days_ago=40)
        finally:
            _restore_env("EMBEDDING_ENABLED", self._prev_emb)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _retrieve_ids(self, expand: bool) -> list[str]:
        # inflight cache key 不含 expansion 状态——A/B 之间必须清空、否则 OFF 会命中 ON 的结果
        with self._store._recall_inflight_lock:
            self._store._recall_inflight_cache.clear()
        with _env("SEMANTIC_QUERY_EXPAND", "1" if expand else "0"):
            prev = os.environ.get("EMBEDDING_ENABLED")
            os.environ["EMBEDDING_ENABLED"] = "true"
            try:
                res = self._store.retrieve_memory_context(
                    "张三", "演唱会安排", query_bundle={}, max_items=5,
                    touch_selected=False, log_retrieval=False,
                )
            finally:
                _restore_env("EMBEDDING_ENABLED", prev)
        return [str(r.get("item_id", "") or "") for r in res.get("rows", [])]

    def test_non_explicit_query_surfaces_old_target_only_with_expansion(self):
        if not getattr(self, "_emb_ok", False):
            self.skipTest("embedding engine unavailable")
        on_ids = self._retrieve_ids(expand=True)
        off_ids = self._retrieve_ids(expand=False)
        self.assertIn("old_target", on_ids,
                      f"expansion ON should surface old_target, got {on_ids}")
        self.assertNotIn("old_target", off_ids,
                         f"expansion OFF should not surface old_target, got {off_ids}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
