"""memory_admin.py — 显式记忆管理 CLI（P0：docs/letta_mem0_memory_plan/07 §4）。

输出协议吸收 mem0 CLI agent-mode：stdout 恒为 JSON 信封
`{status, command, duration_ms, count, data}`；错误同样走 JSON + 非零退出码，
方便将来被 agent 工具循环消费。`--pretty` 给人看的缩进版。

直 import memory_api（不走 HTTP）。一律用 `py -X utf8` 运行：

  py -X utf8 scripts/memory_admin.py list --char 角色A --layer semantic
  py -X utf8 scripts/memory_admin.py search 咖啡 --char 角色A
  py -X utf8 scripts/memory_admin.py get <item_id> --char 角色A
  py -X utf8 scripts/memory_admin.py history <item_id> --char 角色A
  py -X utf8 scripts/memory_admin.py budget --char 角色A --days 7
  py -X utf8 scripts/memory_admin.py core-summary get --char 角色A
  py -X utf8 scripts/memory_admin.py core-summary update --char 角色A --text "..."
  py -X utf8 scripts/memory_admin.py entity list --char 角色A
  py -X utf8 scripts/memory_admin.py entity links --char 角色A --entity-id <entity_id>
  py -X utf8 scripts/memory_admin.py forgetting preview --char 角色A
  py -X utf8 scripts/memory_admin.py forgetting advance --char 角色A --yes
  py -X utf8 scripts/memory_admin.py update <item_id> --char 角色A --text "..." [--reason ...]
  py -X utf8 scripts/memory_admin.py soft-delete <item_id> --char 角色A --yes [--reason ...]
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _emit(command: str, started: float, *, status: str, data, error: str = "", pretty: bool = False) -> int:
    count = len(data) if isinstance(data, list) else (1 if data else 0)
    envelope = {
        "status": status,
        "command": command,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "count": count,
        "data": data,
    }
    if error:
        envelope["error"] = error
    _tsz_emit(json.dumps(envelope, ensure_ascii=False, indent=2 if pretty else None))
    return 0 if status == "ok" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_admin", description="显式记忆管理 CLI（P0）")
    parser.add_argument("--pretty", action="store_true", help="缩进 JSON 输出（给人看）")
    sub = parser.add_subparsers(dest="command", required=True)

    def _char_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--char", required=True, help="角色名")

    p_list = sub.add_parser("list", help="按角色列出记忆")
    _char_arg(p_list)
    p_list.add_argument("--layer", default="", help="working/episodic/semantic/procedural/collective")
    p_list.add_argument("--item-type", default="", help="按 item_type 精确过滤")
    p_list.add_argument("--since", default="", help="created_at 起（ISO）")
    p_list.add_argument("--until", default="", help="created_at 止（ISO）")
    p_list.add_argument("--archived", action="store_true", help="含已归档")
    p_list.add_argument("--collective", action="store_true", help="含集体记忆")
    p_list.add_argument("--limit", type=int, default=200)
    p_list.add_argument("--offset", type=int, default=0)

    p_search = sub.add_parser("search", help="子串搜索记忆文本")
    p_search.add_argument("query")
    _char_arg(p_search)
    p_search.add_argument("--archived", action="store_true", help="含已归档")
    p_search.add_argument("--limit", type=int, default=50)

    p_get = sub.add_parser("get", help="单条详情")
    p_get.add_argument("item_id")
    _char_arg(p_get)

    p_hist = sub.add_parser("history", help="单条变更历史")
    p_hist.add_argument("item_id")
    _char_arg(p_hist)

    p_members = sub.add_parser("members", help="cluster member 展开回读(event/topic_cluster 行)")
    p_members.add_argument("item_id")
    _char_arg(p_members)
    p_members.add_argument("--limit", type=int, default=100)

    p_budget = sub.add_parser("budget", help="per-char/per-day 预算计数")
    _char_arg(p_budget)
    p_budget.add_argument("--days", type=int, default=7)

    p_core_summary = sub.add_parser("core-summary", help="查看或更新 per-character living summary")
    p_core_summary.add_argument("action", choices=("get", "update"))
    _char_arg(p_core_summary)
    p_core_summary.add_argument("--text", default="")
    p_core_summary.add_argument("--reason", default="")

    p_entity = sub.add_parser("entity", help="查看派生 entity 与关联记忆")
    p_entity.add_argument("action", choices=("list", "get", "links"))
    _char_arg(p_entity)
    p_entity.add_argument("--entity-id", default="")
    p_entity.add_argument("--entity-type", default="")
    p_entity.add_argument("--limit", type=int, default=50)

    p_forgetting = sub.add_parser("forgetting", help="预览或推进 Phase E/P4 遗忘生命周期")
    p_forgetting.add_argument("action", choices=("preview", "advance"))
    _char_arg(p_forgetting)
    p_forgetting.add_argument("--frozen-days", type=float, default=90.0)
    p_forgetting.add_argument("--tombstone-days", type=float, default=180.0)
    p_forgetting.add_argument("--yes", action="store_true", help="确认真实推进（仍需 MEM_FORGETTING_APPLY=1）")

    p_update = sub.add_parser("update", help="纠错改写（仅 active 行）")
    p_update.add_argument("item_id")
    _char_arg(p_update)
    p_update.add_argument("--text", required=True)
    p_update.add_argument("--reason", default="")

    p_del = sub.add_parser("soft-delete", help="软删除（状态翻转，绝无物理删除；需 MEM_RETHINK_ALLOW_ARCHIVE=1）")
    p_del.add_argument("item_id")
    _char_arg(p_del)
    p_del.add_argument("--yes", action="store_true", help="确认执行")
    p_del.add_argument("--reason", default="")

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    started = time.monotonic()
    os.chdir(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)

    from tsuzuri.memory_api import get_memory_api  # noqa: E402  （需在 chdir/sys.path 之后）

    api = get_memory_api()
    cmd = str(args.command)

    try:
        if cmd == "list":
            result = api.list_memories(
                args.char,
                layer=args.layer,
                item_type=args.item_type,
                since=args.since,
                until=args.until,
                include_archived=bool(args.archived),
                include_collective=bool(args.collective),
                limit=args.limit,
                offset=args.offset,
            )
            if not result.get("ok"):
                return _emit(cmd, started, status="error", data=[], error=str(result.get("error", "")), pretty=args.pretty)
            return _emit(cmd, started, status="ok", data=result.get("items", []), pretty=args.pretty)

        if cmd == "search":
            result = api.list_memories(
                args.char, query=args.query, include_archived=bool(args.archived), limit=args.limit,
            )
            return _emit(cmd, started, status="ok", data=result.get("items", []), pretty=args.pretty)

        if cmd == "get":
            item = api.get_memory(args.char, args.item_id)
            if item is None:
                return _emit(cmd, started, status="error", data=None, error="not_found", pretty=args.pretty)
            return _emit(cmd, started, status="ok", data=item, pretty=args.pretty)

        if cmd == "history":
            history = api.get_history(args.char, args.item_id)
            if history is None:
                return _emit(cmd, started, status="error", data=None, error="not_found", pretty=args.pretty)
            return _emit(cmd, started, status="ok", data=history.get("entries", []), pretty=args.pretty)

        if cmd == "members":
            result = api.get_cluster_members(args.char, args.item_id, limit=args.limit)
            if result is None:
                return _emit(cmd, started, status="error", data=None, error="not_found", pretty=args.pretty)
            if not result.get("ok"):
                return _emit(cmd, started, status="error", data=[], error=str(result.get("error", "")), pretty=args.pretty)
            return _emit(cmd, started, status="ok", data={
                "cluster": {"item_id": result["cluster"]["item_id"], "title": (result["cluster"].get("payload") or {}).get("title", "")},
                "member_count": result["member_count"],
                "members": result["members"],
            }, pretty=args.pretty)

        if cmd == "budget":
            result = api.get_budget(args.char, days=args.days)
            if not result.get("ok"):
                return _emit(cmd, started, status="error", data=None, error=str(result.get("error", "")), pretty=args.pretty)
            return _emit(cmd, started, status="ok",
                         data={"today": result.get("today", {}), "history": result.get("history", [])},
                         pretty=args.pretty)

        if cmd == "core-summary":
            if args.action == "get":
                result = api.list_memories(args.char, item_type="core_summary", limit=1)
                items = result.get("items", []) if result.get("ok") else []
                if not items:
                    return _emit(cmd, started, status="error", data=None, error="not_found", pretty=args.pretty)
                return _emit(cmd, started, status="ok", data=items[0], pretty=args.pretty)
            if not str(args.text or "").strip():
                return _emit(cmd, started, status="error", data=None, error="text_required", pretty=args.pretty)
            result = api.update_core_summary(
                args.char,
                args.text,
                reason=args.reason,
                actor="dev_cli",
            )
            status = "ok" if result.get("ok") else "error"
            return _emit(
                cmd,
                started,
                status=status,
                data=result,
                error="" if result.get("ok") else str(result.get("error", "")),
                pretty=args.pretty,
            )

        if cmd == "entity":
            if args.action == "list":
                result = api.list_entities(
                    args.char,
                    entity_type=args.entity_type,
                    limit=args.limit,
                )
                status = "ok" if result.get("ok") else "error"
                return _emit(
                    cmd,
                    started,
                    status=status,
                    data=result.get("entities", []),
                    error="" if result.get("ok") else str(result.get("error", "")),
                    pretty=args.pretty,
                )
            if not str(args.entity_id or "").strip():
                return _emit(cmd, started, status="error", data=None, error="entity_id_required", pretty=args.pretty)
            if args.action == "get":
                item = api.get_entity(args.entity_id, char=args.char)
                if item is None:
                    return _emit(cmd, started, status="error", data=None, error="not_found", pretty=args.pretty)
                return _emit(cmd, started, status="ok", data=item, pretty=args.pretty)
            result = api.entity_links(args.entity_id, char=args.char, limit=args.limit)
            status = "ok" if result.get("ok") else "error"
            return _emit(
                cmd,
                started,
                status=status,
                data=result.get("links", []),
                error="" if result.get("ok") else str(result.get("error", "")),
                pretty=args.pretty,
            )

        if cmd == "forgetting":
            apply = args.action == "advance"
            if apply and not bool(args.yes):
                return _emit(
                    cmd,
                    started,
                    status="error",
                    data=None,
                    error="confirm_required: advance 需加 --yes",
                    pretty=args.pretty,
                )
            result = api.advance_forgetting(
                args.char,
                dry_run=not apply,
                frozen_min_days=args.frozen_days,
                tombstone_min_days=args.tombstone_days,
                actor="dev_cli",
            )
            status = "ok" if result.get("ok") else "error"
            return _emit(
                cmd,
                started,
                status=status,
                data=result,
                error="" if result.get("ok") else str(result.get("error", "")),
                pretty=args.pretty,
            )

        if cmd == "update":
            result = api.update_memory(args.char, args.item_id, args.text, reason=args.reason, actor="dev_cli")
            status = "ok" if result.get("ok") else "error"
            return _emit(cmd, started, status=status, data=result,
                         error="" if result.get("ok") else str(result.get("error", "")), pretty=args.pretty)

        if cmd == "soft-delete":
            if not bool(args.yes):
                return _emit(cmd, started, status="error", data=None, error="confirm_required: 加 --yes 确认", pretty=args.pretty)
            result = api.delete_memory(args.char, args.item_id, reason=args.reason, actor="dev_cli")
            status = "ok" if result.get("ok") else "error"
            err = "" if result.get("ok") else str(result.get("error", ""))
            if result.get("detail"):
                err = f"{err}: {result['detail']}"
            return _emit(cmd, started, status=status, data=result, error=err, pretty=args.pretty)

        return _emit(cmd, started, status="error", data=None, error=f"unknown_command:{cmd}", pretty=args.pretty)
    except Exception as exc:  # 信封永远落 stdout，异常不裸奔
        return _emit(cmd, started, status="error", data=None, error=str(exc)[:300], pretty=args.pretty)


if __name__ == "__main__":
    sys.exit(main())
