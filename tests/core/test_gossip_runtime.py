# -*- coding: utf-8 -*-
"""gossip_runtime 单测（2026-08-14 gossip 链路统一）。

覆盖此前零测试的传播数学（confidence 公式、fanout 概率、hop 上限、rng 注入）
与 shared_info 统一存储层行为（record_fanout / record_secondary /
record_town_mailbox / get_pending_for / mark_delivered）。
存储测试用临时库，不触碰项目真实 memory_core.db。
"""

import json
import time
import unittest

from tsuzuri import gossip_runtime as gossip_runtime


class GossipProfileTests(unittest.TestCase):
    def test_profile_covers_five_chars(self):
        self.assertEqual(set(gossip_runtime.GOSSIP_PROFILE), {"张三", "赵六", "李四", "王五", "钱七"})

    def test_unknown_char_fallback(self):
        p = gossip_runtime.gossip_profile("不存在")
        self.assertAlmostEqual(p["spread_prob"], 0.3)
        self.assertAlmostEqual(p["distortion"], 0.1)

    def test_delay_ticks_removed(self):
        for char, profile in gossip_runtime.GOSSIP_PROFILE.items():
            self.assertNotIn("delay_ticks", profile, char)


class GossipMathTests(unittest.TestCase):
    def test_first_hop_confidence_formula(self):
        # 李四 distortion 0.15，trust 0.8 → 1 - .15 - .1*(1-.8) = 0.83
        self.assertAlmostEqual(gossip_runtime.first_hop_confidence("李四", 0.8), 0.83)
        # 钱七 distortion 0，trust 0 → 1 - 0 - .1 = 0.9
        self.assertAlmostEqual(gossip_runtime.first_hop_confidence("钱七", 0.0), 0.9)
        # 下限 0.3
        self.assertEqual(gossip_runtime.first_hop_confidence("李四", -5.0), 0.3)

    def test_next_hop_confidence_decay(self):
        self.assertAlmostEqual(gossip_runtime.next_hop_confidence(0.9), 0.63)
        self.assertAlmostEqual(gossip_runtime.next_hop_confidence(0.63), 0.44)

    def test_spread_probability(self):
        edge = {"info_sharing": 0.4}
        self.assertAlmostEqual(gossip_runtime.spread_probability("张三", edge), 0.34)
        # 2026-08-27 §9-3：secondary 衰减系数 0.5 → 0.7
        self.assertAlmostEqual(
            gossip_runtime.spread_probability("张三", edge, secondary=True),
            0.34 * gossip_runtime.SECONDARY_PROB_MULTIPLIER,
        )
        self.assertAlmostEqual(gossip_runtime.spread_probability("未知角色", {}), 0.3 * 0.3)

    def test_roll_spread_deterministic_with_injected_rng(self):
        edge = {"info_sharing": 1.0}
        # 张三 spread_prob 0.85：roll 0.84 传播，0.86 不传播
        self.assertTrue(gossip_runtime.roll_spread("张三", edge, rng=lambda: 0.84))
        self.assertFalse(gossip_runtime.roll_spread("张三", edge, rng=lambda: 0.86))
        # 钱七 0.20：中值 roll 不传播
        self.assertFalse(gossip_runtime.roll_spread("钱七", edge, rng=lambda: 0.5))


pass  # upstream-only test removed


class GossipStoreTests(unittest.TestCase):
    """shared_info 统一存储层行为（临时库）。"""

    def setUp(self):
        import os
        import tempfile
        from tsuzuri.memory_core import MemoryCoreStore

        self._td = tempfile.TemporaryDirectory()
        snap = os.path.join(self._td.name, "snap.json")
        with open(snap, "w", encoding="utf-8") as f:
            f.write("{}")
        self.store = MemoryCoreStore(db_path=os.path.join(self._td.name, "m.db"), snapshot_path=snap)
        self.store.ensure_shared_info_schema()
        # mailbox 投影的进程内去重缓存是模块级状态，测试间必须复位
        gossip_runtime._TOWN_MAILBOX_SETTLED.clear()
        gossip_runtime._TOWN_MAILBOX_TEXT_SETTLED.clear()

    def tearDown(self):
        self._td.cleanup()

    def _audit_rows(self, event_kind=None):
        import sqlite3
        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if event_kind:
                rows = conn.execute(
                    "SELECT * FROM gossip_audit_log WHERE event_kind=? ORDER BY log_id;",
                    (event_kind,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM gossip_audit_log ORDER BY log_id;").fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def test_fanout_writes_per_target_rows_with_eligibility(self):
        graph = {"张三": {"李四": {"trust": 0.8, "info_sharing": 1.0}}}
        ids = gossip_runtime.record_fanout("张三", "阿澈说他今天去看了天文馆", graph=graph, rng=lambda: 0.01, store=self.store)
        self.assertEqual(len(ids), 4)  # 除自己外四个目标
        pend = gossip_runtime.get_pending_for("李四", store=self.store)
        self.assertEqual(len(pend), 1)
        self.assertEqual(pend[0]["source_character"], "张三")
        self.assertAlmostEqual(pend[0]["confidence"], 0.88)  # trust .8 → 1-.10-.02
        # 源本人不可见自己的 gossip
        self.assertEqual(gossip_runtime.get_pending_for("张三", store=self.store), [])

    def test_fanout_rng_all_miss_writes_nothing(self):
        ids = gossip_runtime.record_fanout("张三", "一条传闻内容", graph={}, rng=lambda: 0.99, store=self.store)
        self.assertEqual(ids, [])
        self.assertEqual(gossip_runtime.get_pending_for("李四", store=self.store), [])

    def test_validator_drops_stranger_framing_and_fixes_terms(self):
        ids = gossip_runtime.record_fanout("张三", "今天认识了一个叫阿澈的人", graph={}, rng=lambda: 0.0, store=self.store)
        self.assertEqual(ids, [])
        ids = gossip_runtime.record_fanout("张三", "用户说这周三要加班开会", graph={}, rng=lambda: 0.0, store=self.store)
        self.assertTrue(ids)
        text = gossip_runtime.get_pending_for("李四", store=self.store)[0]["original_text"]
        self.assertIn("阿澈", text)
        self.assertNotIn("用户", text)

    def test_mark_delivered_hides_row(self):
        gossip_runtime.record_fanout("张三", "阿澈说他今天去看展", graph={}, rng=lambda: 0.0, store=self.store)
        pend = gossip_runtime.get_pending_for("赵六", store=self.store)
        self.assertEqual(len(pend), 1)
        gossip_runtime.mark_delivered(pend[0]["info_id"], "赵六", store=self.store)
        self.assertEqual(gossip_runtime.get_pending_for("赵六", store=self.store), [])

    def test_secondary_requires_a_actually_received(self):
        # 赵六 → 李四（李四 eligible）
        gossip_runtime.record_fanout("赵六", "赵六的传闻内容一二三", graph={"赵六": {"李四": {"trust": 0.9, "info_sharing": 1.0}}}, rng=lambda: 0.01, store=self.store)
        # 李四还没收到（未 mark_delivered）→ 李四不能转发给王五（修复旧语义 bug）
        ok = gossip_runtime.record_secondary("李四", "王五", graph={"李四": {"王五": {"info_sharing": 1.0}}}, rng=lambda: 0.0, store=self.store)
        self.assertFalse(ok)
        # 李四收到后可以转发，hop+1、confidence ×0.7
        pend = gossip_runtime.get_pending_for("李四", store=self.store)
        gossip_runtime.mark_delivered(pend[0]["info_id"], "李四", store=self.store)
        ok = gossip_runtime.record_secondary("李四", "王五", graph={"李四": {"王五": {"info_sharing": 1.0}}}, rng=lambda: 0.0, store=self.store)
        self.assertTrue(ok)
        rows = [r for r in gossip_runtime.get_pending_for("王五", store=self.store) if r["source_character"] == "李四"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hop"], 2)
        self.assertAlmostEqual(rows[0]["confidence"], round(pend[0]["confidence"] * 0.7, 2))

    def test_secondary_hop_cap(self):
        gossip_runtime.record_fanout("赵六", "多手传闻内容一二三", graph={"赵六": {"李四": {"trust": 0.9, "info_sharing": 1.0}}}, rng=lambda: 0.01, store=self.store)
        pend = gossip_runtime.get_pending_for("李四", store=self.store)
        gossip_runtime.mark_delivered(pend[0]["info_id"], "李四", store=self.store)
        # 手动把该行 hop 抬到 MAX_HOP → 不再二次传播
        import sqlite3
        conn = sqlite3.connect(self.store.db_path)
        conn.execute("UPDATE shared_info SET hop=? WHERE info_id=?", (gossip_runtime.MAX_HOP, pend[0]["info_id"]))
        conn.commit()
        conn.close()
        ok = gossip_runtime.record_secondary("李四", "王五", graph={"李四": {"王五": {"info_sharing": 1.0}}}, rng=lambda: 0.0, store=self.store)
        self.assertFalse(ok)

    def test_town_mailbox_projection_dedup(self):
        items = [{"id": "m1", "from": "李四", "topic": "天文馆", "summary": "下周再去一次"}]
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 1)
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 0)  # 幂等
        rows = gossip_runtime.get_pending_for("钱七", store=self.store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hop"], gossip_runtime.MAILBOX_LAYER_HOP)
        # 2026-10-05：摘要 confidence 对齐来源 agent_message 事件（不再高于真八卦首跳），
        # 但 0.70 仍落在 hearsay box 第一档 "{from}说：" 的阈值上。
        self.assertEqual(rows[0]["confidence"], gossip_runtime.MAILBOX_PROJECTION_CONFIDENCE)
        self.assertLess(rows[0]["confidence"], 1.0)
        self.assertIn("天文馆", rows[0]["original_text"])

    def test_recent_mailbox_remains_high_coverage_for_low_spread_character(self):
        """近期对话摘要不受来源角色的 fanout 概率限制。"""
        item = {"id": "recent-low-spread", "from": "钱七", "topic": "集训安排", "summary": "明天改到下午"}
        self.assertEqual(gossip_runtime.record_town_mailbox("李四", [item], store=self.store), 1)
        rows = gossip_runtime.get_pending_for("李四", store=self.store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hop"], gossip_runtime.MAILBOX_LAYER_HOP)
        self.assertEqual(rows[0]["source_character"], "钱七")

    def test_member_gossip_frequency_still_uses_spread_probability(self):
        """成员间传闻继续保留低 spread_prob 的频率差异。"""
        ids = gossip_runtime.record_fanout(
            "钱七",
            "阿澈说他明天要去看天文馆",
            graph={name: {"李四": {"trust": 0.8, "info_sharing": 1.0}} for name in ("钱七",)},
            rng=lambda: 0.5,
            store=self.store,
        )
        self.assertEqual(ids, [])
        self.assertEqual(gossip_runtime.get_pending_for("李四", store=self.store), [])

    def test_town_mailbox_content_dedup_across_message_ids(self):
        # town_sim 近况同步事件的 idempotency_key 含 tick id：同一内容每 tick 换新 eid，
        # (town_msg_id, character) 去重拦不住，会把读取窗口挤满（2026-10-05 审计）。
        base = {"from": "李四", "topic": "天文馆", "summary": "下周再去一次"}
        for tick in range(5):
            self.assertEqual(
                gossip_runtime.record_town_mailbox(
                    "钱七", [dict(base, id=f"eid-tick-{tick}")], store=self.store,
                ),
                1 if tick == 0 else 0,
            )
        rows = gossip_runtime.get_pending_for("钱七", store=self.store, limit=10)
        self.assertEqual(len(rows), 1)
        attempts = self._audit_rows("mailbox_attempt")
        self.assertEqual(len(attempts), 5)  # 审计量不变：仍是每次 sync 一条
        self.assertEqual(json.loads(attempts[-1]["detail_json"])["content_dedup_skipped"], 1)
        # 同内容投给别的角色仍然成立（去重按 角色+文本，不是全局）
        self.assertEqual(
            gossip_runtime.record_town_mailbox(
                "王五", [dict(base, id="eid-other-target")], store=self.store,
            ),
            1,
        )

    def test_town_mailbox_content_dedup_within_single_batch(self):
        # 同一次 sync 的 inbox_digest 带多条同文本消息时，DB 兜底还看不到彼此，
        # 必须靠批内去重；prompt 构建期的投影（testgroup._build_town_projection_prompt_block
        # → sync_projection_to_legacy）实测就是这个形态。
        item = {"from": "李四", "topic": "天文馆", "summary": "下周再去一次"}
        added = gossip_runtime.record_town_mailbox(
            "钱七",
            [dict(item, id=f"eid-batch-{i}") for i in range(3)],
            store=self.store,
        )
        self.assertEqual(added, 1)
        rows = gossip_runtime.get_pending_for("钱七", store=self.store, limit=10)
        self.assertEqual(len(rows), 1)
        attempt = self._audit_rows("mailbox_attempt")[-1]
        detail = json.loads(attempt["detail_json"])
        self.assertEqual(detail["items"], 3)
        self.assertEqual(detail["content_dedup_skipped"], 2)

    def test_town_mailbox_content_dedup_survives_cache_reset(self):
        base = {"id": "eid-a", "from": "李四", "topic": "天文馆", "summary": "下周再去一次"}
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", [base], store=self.store), 1)
        # 模拟进程重启：清进程内缓存后由 DB 兜底拦下同内容换 id 的重投
        gossip_runtime._TOWN_MAILBOX_SETTLED.clear()
        gossip_runtime._TOWN_MAILBOX_TEXT_SETTLED.clear()
        self.assertEqual(
            gossip_runtime.record_town_mailbox(
                "钱七", [dict(base, id="eid-b")], store=self.store,
            ),
            0,
        )
        self.assertEqual(len(self._audit_rows("mailbox_projected")), 1)

    def test_pending_quota_keeps_real_gossip_visible_under_digest_flood(self):
        # 2026-10-05 审计根因：hop0 摘要（约 32 行/日）按 created_at DESC 把 hop1
        # 真八卦（约 7 行/日）挤出读取窗口，导致 42% 的 hop1 行零投递过期。
        now = time.time()
        self.store.store_shared_info(
            info_id="si-real-gossip", source_character="张三",
            original_text="阿澈说他这周要去看天文馆", created_at=now - 6 * 3600,
            confidence=0.88, hop=1, eligible_targets=["李四"],
        )
        for i in range(12):  # 比真八卦更新的大量摘要
            self.store.store_shared_info(
                info_id=f"si-digest-{i}", source_character="王五",
                original_text=f"近况同步：王五在教室推进了整理{i}号", created_at=now - i * 60,
                confidence=gossip_runtime.MAILBOX_PROJECTION_CONFIDENCE,
                hop=gossip_runtime.MAILBOX_LAYER_HOP, eligible_targets=["李四"],
            )
        rows = gossip_runtime.get_pending_for("李四", limit=6, store=self.store)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["info_id"], "si-real-gossip")  # 人际层优先
        self.assertEqual([r["hop"] for r in rows].count(1), 1)
        self.assertEqual([r["hop"] for r in rows].count(0), 5)
        # hearsay box 只保留 2 条：分层后第 1 条必定是真八卦
        top2 = gossip_runtime.get_pending_for("李四", limit=2, store=self.store)
        self.assertEqual(top2[0]["info_id"], "si-real-gossip")
        self.assertEqual(len(top2), 2)

    def test_pending_quota_backfills_when_one_layer_is_empty(self):
        now = time.time()
        for i in range(4):  # 只有摘要层
            self.store.store_shared_info(
                info_id=f"si-digest-only-{i}", source_character="王五",
                original_text=f"近况同步：王五在教室推进了整理{i}号", created_at=now - i * 60,
                confidence=gossip_runtime.MAILBOX_PROJECTION_CONFIDENCE,
                hop=gossip_runtime.MAILBOX_LAYER_HOP, eligible_targets=["李四"],
            )
        rows = gossip_runtime.get_pending_for("李四", limit=3, store=self.store)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["hop"] == gossip_runtime.MAILBOX_LAYER_HOP for r in rows))

        for i in range(4):  # 只有人际层
            self.store.store_shared_info(
                info_id=f"si-real-only-{i}", source_character="张三",
                original_text=f"阿澈说他这周要去看天文馆{i}号", created_at=now - i * 60,
                confidence=0.88, hop=1, eligible_targets=["赵六"],
            )
        rows = gossip_runtime.get_pending_for("赵六", limit=3, store=self.store)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["hop"] >= 1 for r in rows))

    def test_pending_slots_split(self):
        self.assertEqual(gossip_runtime._pending_slots(1), (1, 0))
        self.assertEqual(gossip_runtime._pending_slots(2), (1, 1))
        self.assertEqual(gossip_runtime._pending_slots(3), (2, 1))
        self.assertEqual(gossip_runtime._pending_slots(6), (3, 3))
        for limit in (1, 2, 3, 6, 10):
            real, digest = gossip_runtime._pending_slots(limit)
            self.assertEqual(real + digest, limit)
            self.assertGreaterEqual(real, 1)

    def test_freshness_window_expires_old_rows(self):
        gossip_runtime.record_fanout("张三", "阿澈说他今天去看展", graph={}, rng=lambda: 0.0, store=self.store)
        future = __import__("time").time() + 49 * 3600  # 48h 窗口之后
        self.assertEqual(gossip_runtime.get_pending_for("李四", store=self.store, now_ts=future), [])

    def test_audit_log_records_fanout_hits_misses_and_blocked(self):
        gossip_runtime.record_fanout("张三", "阿澈说他今天去看展", graph={}, rng=lambda: 0.0, store=self.store)
        hits = self._audit_rows("fanout_hit")
        self.assertEqual(len(hits), 4)
        attempts = self._audit_rows("fanout_attempt")
        self.assertEqual(len(attempts), 1)
        self.assertFalse(json.loads(attempts[0]["detail_json"])["all_miss"])
        self.assertEqual(json.loads(attempts[0]["detail_json"])["hit_count"], 4)

        gossip_runtime.record_fanout("张三", "一条不会传播的传闻", graph={}, rng=lambda: 0.99, store=self.store)
        self.assertEqual(len(self._audit_rows("fanout_miss")), 4)
        attempts = self._audit_rows("fanout_attempt")
        self.assertEqual(len(attempts), 2)
        self.assertTrue(json.loads(attempts[1]["detail_json"])["all_miss"])

        gossip_runtime.record_blocked("李四", "阿澈说他想看星星", reason="gossip_write_blocked", store=self.store)
        gossip_runtime.record_fanout("张三", "今天认识了一个叫阿澈的人", graph={}, rng=lambda: 0.0, store=self.store)
        blocked = self._audit_rows("fanout_blocked")
        reasons = {json.loads(r["detail_json"]).get("reason") for r in blocked}
        self.assertIn("gossip_write_blocked", reasons)
        self.assertIn("validator_drop", reasons)

    def test_audit_log_records_delivery_secondary_and_mailbox(self):
        gossip_runtime.record_fanout(
            "赵六", "赵六的传闻内容一二三",
            graph={"赵六": {"李四": {"trust": 0.9, "info_sharing": 1.0}}},
            rng=lambda: 0.01, store=self.store,
        )
        pend = gossip_runtime.get_pending_for("李四", store=self.store)
        gossip_runtime.mark_delivered(pend[0]["info_id"], "李四", store=self.store)
        delivered = self._audit_rows("delivered")
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]["target_character"], "李四")

        ok = gossip_runtime.record_secondary(
            "李四", "王五",
            graph={"李四": {"王五": {"info_sharing": 1.0}}},
            rng=lambda: 0.0, store=self.store,
        )
        self.assertTrue(ok)
        written = self._audit_rows("secondary_written")
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["source_character"], "李四")
        self.assertEqual(written[0]["target_character"], "王五")

        items = [{"id": "m1", "from": "李四", "topic": "天文馆", "summary": "下周再去一次"}]
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 1)
        # 模拟进程重启：清进程内缓存，第二次由 DB 兜底去重拦下；
        # 兜底命中静默跳过，不再写 mailbox_skip_duplicate 审计事件
        gossip_runtime._TOWN_MAILBOX_SETTLED.clear()
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 0)
        projected = self._audit_rows("mailbox_projected")
        skipped = self._audit_rows("mailbox_skip_duplicate")
        self.assertEqual(len(projected), 1)
        self.assertEqual(len(skipped), 0)
        self.assertEqual(projected[0]["target_character"], "钱七")

    def test_extract_audit_accumulates_daily_counts_and_validator_drop(self):
        gossip_runtime.record_extract_audit("李四", has_gossip=True, store=self.store)
        gossip_runtime.record_extract_audit("李四", has_gossip=True, blocked_gate=True, store=self.store)
        gossip_runtime.record_extract_audit("张三", has_gossip=False, store=self.store)
        gossip_runtime.record_fanout("张三", "今天认识了一个叫阿澈的人", graph={}, rng=lambda: 0.0, store=self.store)

        import sqlite3
        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM gossip_extract_audit ORDER BY character;").fetchall()
        finally:
            conn.close()
        by_char = {r["character"]: dict(r) for r in rows}
        self.assertEqual(by_char["李四"]["extractions"], 2)
        self.assertEqual(by_char["李四"]["gossip_outputs"], 2)
        self.assertEqual(by_char["李四"]["blocked_gate"], 1)
        self.assertEqual(by_char["张三"]["extractions"], 1)
        self.assertEqual(by_char["张三"]["gossip_outputs"], 0)
        self.assertEqual(by_char["张三"]["validator_drop"], 1)

    def test_town_mailbox_cache_hit_skips_audit_noise(self):
        # 2026-08-27 §9-2：进程内缓存命中后不再写 mailbox_attempt / skip_duplicate 审计。
        items = [{"id": "m-cache", "from": "李四", "topic": "天文馆", "summary": "下周再去一次"}]
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 1)
        attempts_after_first = len(self._audit_rows("mailbox_attempt"))
        self.assertEqual(gossip_runtime.record_town_mailbox("钱七", items, store=self.store), 0)
        self.assertEqual(len(self._audit_rows("mailbox_attempt")), attempts_after_first)
        self.assertEqual(len(self._audit_rows("mailbox_skip_duplicate")), 0)

    def test_secondary_tries_up_to_max_candidates_per_tick(self):
        # 2026-08-27 §9-3：每 tick 最多试 SECONDARY_MAX_TRIES_PER_TICK 条，成功即止。
        now = time.time()
        for i in range(4):
            self.store.store_shared_info(
                info_id=f"si-sec-{i}", source_character="赵六",
                original_text=f"赵六的候选传闻{i}号", created_at=now,
                eligible_targets=["李四"],
            )
            self.store.mark_shared_info_delivered(f"si-sec-{i}", "李四", delivered_at=now)

        ok = gossip_runtime.record_secondary(
            "李四", "王五",
            graph={"李四": {"王五": {"info_sharing": 1.0}}},
            rng=lambda: 0.99, store=self.store,
        )
        self.assertFalse(ok)
        self.assertEqual(len(self._audit_rows("secondary_written")), 0)
        self.assertEqual(
            len(self._audit_rows("secondary_miss")),
            gossip_runtime.SECONDARY_MAX_TRIES_PER_TICK,
        )

    def test_secondary_skips_stale_candidates(self):
        # 2026-08-27 §9-3：超过 48h 新鲜度窗口的候选不转述（写进去也投不出去）。
        old = time.time() - 60 * 3600
        self.store.store_shared_info(
            info_id="si-stale-secondary", source_character="赵六",
            original_text="李四说他上周去了天文馆", created_at=old,
            eligible_targets=["李四"],
        )
        self.store.mark_shared_info_delivered("si-stale-secondary", "李四", delivered_at=time.time())

        ok = gossip_runtime.record_secondary(
            "李四", "王五",
            graph={"李四": {"王五": {"info_sharing": 1.0}}},
            rng=lambda: 0.0, store=self.store,
        )
        self.assertFalse(ok)

    def test_secondary_query_ignores_hop0_digest_flood(self):
        now = time.time()
        self.store.store_shared_info(
            info_id="si-secondary-visible", source_character="赵六",
            original_text="赵六说集训时间改了", created_at=now - 3600,
            eligible_targets=["李四"], hop=1,
        )
        self.store.mark_shared_info_delivered("si-secondary-visible", "李四", delivered_at=now)
        for i in range(20):
            self.store.store_shared_info(
                info_id=f"si-hop0-flood-{i}", source_character="张三",
                original_text=f"近况摘要{i}", created_at=now - i,
                eligible_targets=["李四"], hop=gossip_runtime.MAILBOX_LAYER_HOP,
            )
        ok = gossip_runtime.record_secondary(
            "李四", "王五",
            graph={"李四": {"王五": {"info_sharing": 1.0}}},
            rng=lambda: 0.0, store=self.store,
        )
        self.assertTrue(ok)
        rows = gossip_runtime.get_pending_for("王五", store=self.store, limit=10)
        self.assertTrue(any(row["hop"] == 2 for row in rows))

    def test_secondary_does_not_consider_unheard_hop1(self):
        self.store.store_shared_info(
            info_id="si-secondary-unheard", source_character="赵六",
            original_text="赵六说一条李四没听到的传闻", created_at=time.time() - 3600,
            eligible_targets=["张三"], hop=1,
        )
        ok = gossip_runtime.record_secondary(
            "李四", "王五",
            graph={"李四": {"王五": {"info_sharing": 1.0}}},
            rng=lambda: 0.0, store=self.store,
        )
        self.assertFalse(ok)
        noops = self._audit_rows("secondary_noop")
        self.assertEqual(len(noops), 1)
        self.assertEqual(json.loads(noops[0]["detail_json"]).get("outcome"), "no_received_candidate")
        self.assertEqual(len(self._audit_rows("secondary_written")), 0)


if __name__ == "__main__":
    unittest.main()


pass  # upstream-only test removed
