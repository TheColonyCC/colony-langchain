"""Targeted tests to bring langchain_colony to 100% coverage.

These tests close specific coverage gaps that the broader test files
don't reach — mostly error paths, async paths through the sync client,
and small branches in metadata extraction / lazy imports.

Every test here exists to keep a single line green; if you remove a
test, check the coverage report to confirm what it was protecting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from colony_sdk import ColonyAPIError, ColonyNotFoundError
from colony_sdk.testing import MockColonyClient

from langchain_colony import (
    ColonyCallbackHandler,
    ColonyEventPoller,
    ColonyToolkit,
)
from langchain_colony.callbacks import _extract_metadata
from langchain_colony.retriever import ColonyRetriever

# ── Helpers ──────────────────────────────────────────────────────────


def _raise_not_found(**_kw: Any) -> None:
    raise ColonyNotFoundError("not found", status=404)


def _raise_generic(**_kw: Any) -> None:
    raise RuntimeError("boom")


def _toolkit_with_error(method: str, error_fn: Any = _raise_not_found) -> tuple[ColonyToolkit, MockColonyClient]:
    """Build a toolkit whose given method raises an exception when called."""
    mock = MockColonyClient(responses={method: error_fn})
    return ColonyToolkit(client=mock), mock


# ── tools.py error paths ─────────────────────────────────────────────
#
# Each tool's _run / _arun has the pattern:
#     data = self._api(...) / await self._aapi(...)
#     if isinstance(data, str):
#         return data   # ← uncovered
#     return _format_xxx(data)
#
# We exercise that branch by making the SDK method raise — _api / _aapi
# catches the exception and returns a friendly-error string instead of
# the dict, which trips the isinstance check.


class TestToolsErrorPaths:
    """Cover the `if isinstance(data, str): return data` branch in every tool."""

    def _tool(self, name: str, mock: MockColonyClient) -> Any:
        toolkit = ColonyToolkit(client=mock)
        return next(t for t in toolkit.get_tools() if t.name == name)

    def test_search_posts_error(self) -> None:
        mock = MockColonyClient(responses={"get_posts": _raise_not_found})
        result = self._tool("colony_search_posts", mock).invoke({"query": "x"})
        assert "Error" in result

    def test_search_posts_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_posts": _raise_not_found})
        result = asyncio.run(self._tool("colony_search_posts", mock).ainvoke({"query": "x"}))
        assert "Error" in result

    def test_get_post_error(self) -> None:
        mock = MockColonyClient(responses={"get_post": _raise_not_found})
        result = self._tool("colony_get_post", mock).invoke({"post_id": "p"})
        assert "Error" in result

    def test_get_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_post", mock).ainvoke({"post_id": "p"}))
        assert "Error" in result

    def test_create_post_error(self) -> None:
        mock = MockColonyClient(responses={"create_post": _raise_not_found})
        result = self._tool("colony_create_post", mock).invoke({"title": "t", "body": "b"})
        assert "Error" in result

    def test_create_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"create_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_create_post", mock).ainvoke({"title": "t", "body": "b"}))
        assert "Error" in result

    def test_comment_on_post_error(self) -> None:
        mock = MockColonyClient(responses={"create_comment": _raise_not_found})
        result = self._tool("colony_comment_on_post", mock).invoke({"post_id": "p", "body": "b"})
        assert "Error" in result

    def test_comment_on_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"create_comment": _raise_not_found})
        result = asyncio.run(self._tool("colony_comment_on_post", mock).ainvoke({"post_id": "p", "body": "b"}))
        assert "Error" in result

    def test_vote_on_post_error(self) -> None:
        mock = MockColonyClient(responses={"vote_post": _raise_not_found})
        result = self._tool("colony_vote_on_post", mock).invoke({"post_id": "p", "value": 1})
        assert "Error" in result

    def test_vote_on_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"vote_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_vote_on_post", mock).ainvoke({"post_id": "p", "value": 1}))
        assert "Error" in result

    def test_send_message_error(self) -> None:
        mock = MockColonyClient(responses={"send_message": _raise_not_found})
        result = self._tool("colony_send_message", mock).invoke({"username": "u", "body": "b"})
        assert "Error" in result

    def test_send_message_async_error(self) -> None:
        mock = MockColonyClient(responses={"send_message": _raise_not_found})
        result = asyncio.run(self._tool("colony_send_message", mock).ainvoke({"username": "u", "body": "b"}))
        assert "Error" in result

    def test_get_notifications_error(self) -> None:
        mock = MockColonyClient(responses={"get_notifications": _raise_not_found})
        result = self._tool("colony_get_notifications", mock).invoke({})
        assert "Error" in result

    def test_get_notifications_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_notifications": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_notifications", mock).ainvoke({}))
        assert "Error" in result

    def test_get_me_error(self) -> None:
        mock = MockColonyClient(responses={"get_me": _raise_not_found})
        result = self._tool("colony_get_me", mock).invoke({})
        assert "Error" in result

    def test_get_me_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_me": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_me", mock).ainvoke({}))
        assert "Error" in result

    def test_get_user_error(self) -> None:
        mock = MockColonyClient(responses={"get_user": _raise_not_found})
        result = self._tool("colony_get_user", mock).invoke({"user_id": "u"})
        assert "Error" in result

    def test_get_user_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_user": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_user", mock).ainvoke({"user_id": "u"}))
        assert "Error" in result

    def test_list_colonies_error(self) -> None:
        mock = MockColonyClient(responses={"get_colonies": _raise_not_found})
        result = self._tool("colony_list_colonies", mock).invoke({})
        assert "Error" in result

    def test_list_colonies_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_colonies": _raise_not_found})
        result = asyncio.run(self._tool("colony_list_colonies", mock).ainvoke({}))
        assert "Error" in result

    def test_get_conversation_error(self) -> None:
        mock = MockColonyClient(responses={"get_conversation": _raise_not_found})
        result = self._tool("colony_get_conversation", mock).invoke({"username": "u"})
        assert "Error" in result

    def test_get_conversation_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_conversation": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_conversation", mock).ainvoke({"username": "u"}))
        assert "Error" in result

    def test_get_poll_error(self) -> None:
        mock = MockColonyClient(responses={"get_poll": _raise_not_found})
        result = self._tool("colony_get_poll", mock).invoke({"post_id": "p"})
        assert "Error" in result

    def test_get_poll_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_poll": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_poll", mock).ainvoke({"post_id": "p"}))
        assert "Error" in result

    def test_get_webhooks_error(self) -> None:
        mock = MockColonyClient(responses={"get_webhooks": _raise_not_found})
        result = self._tool("colony_get_webhooks", mock).invoke({})
        assert "Error" in result

    def test_get_webhooks_async_error(self) -> None:
        mock = MockColonyClient(responses={"get_webhooks": _raise_not_found})
        result = asyncio.run(self._tool("colony_get_webhooks", mock).ainvoke({}))
        assert "Error" in result

    def test_update_post_error(self) -> None:
        mock = MockColonyClient(responses={"update_post": _raise_not_found})
        result = self._tool("colony_update_post", mock).invoke({"post_id": "p", "title": "t"})
        assert "Error" in result

    def test_update_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"update_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_update_post", mock).ainvoke({"post_id": "p", "title": "t"}))
        assert "Error" in result

    def test_delete_post_error(self) -> None:
        mock = MockColonyClient(responses={"delete_post": _raise_not_found})
        result = self._tool("colony_delete_post", mock).invoke({"post_id": "p"})
        assert "Error" in result

    def test_delete_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"delete_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_delete_post", mock).ainvoke({"post_id": "p"}))
        assert "Error" in result

    def test_vote_on_comment_error(self) -> None:
        mock = MockColonyClient(responses={"vote_comment": _raise_not_found})
        result = self._tool("colony_vote_on_comment", mock).invoke({"comment_id": "c", "value": 1})
        assert "Error" in result

    def test_vote_on_comment_async_error(self) -> None:
        mock = MockColonyClient(responses={"vote_comment": _raise_not_found})
        result = asyncio.run(self._tool("colony_vote_on_comment", mock).ainvoke({"comment_id": "c", "value": 1}))
        assert "Error" in result

    def test_mark_notifications_read_error(self) -> None:
        mock = MockColonyClient(responses={"mark_notifications_read": _raise_not_found})

        # mark_notifications_read returns None on the mock — patch the bound method to raise.
        def boom() -> None:
            raise ColonyNotFoundError("nope", status=404)

        mock.mark_notifications_read = boom  # type: ignore[method-assign]
        result = self._tool("colony_mark_notifications_read", mock).invoke({})
        assert "Error" in result

    def test_mark_notifications_read_async_error(self) -> None:
        mock = MockColonyClient()

        def boom() -> None:
            raise ColonyNotFoundError("nope", status=404)

        mock.mark_notifications_read = boom  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_mark_notifications_read", mock).ainvoke({}))
        assert "Error" in result

    def test_update_profile_error(self) -> None:
        mock = MockColonyClient(responses={"update_profile": _raise_not_found})
        result = self._tool("colony_update_profile", mock).invoke({"display_name": "x"})
        assert "Error" in result

    def test_update_profile_async_error(self) -> None:
        mock = MockColonyClient(responses={"update_profile": _raise_not_found})
        result = asyncio.run(self._tool("colony_update_profile", mock).ainvoke({"display_name": "x"}))
        assert "Error" in result

    def test_follow_user_error(self) -> None:
        mock = MockColonyClient(responses={"follow": _raise_not_found})
        result = self._tool("colony_follow_user", mock).invoke({"user_id": "u"})
        assert "Error" in result

    def test_follow_user_async_error(self) -> None:
        mock = MockColonyClient(responses={"follow": _raise_not_found})
        result = asyncio.run(self._tool("colony_follow_user", mock).ainvoke({"user_id": "u"}))
        assert "Error" in result

    def test_unfollow_user_error(self) -> None:
        mock = MockColonyClient(responses={"unfollow": _raise_not_found})
        result = self._tool("colony_unfollow_user", mock).invoke({"user_id": "u"})
        assert "Error" in result

    def test_unfollow_user_async_error(self) -> None:
        mock = MockColonyClient(responses={"unfollow": _raise_not_found})
        result = asyncio.run(self._tool("colony_unfollow_user", mock).ainvoke({"user_id": "u"}))
        assert "Error" in result

    def test_react_to_post_error(self) -> None:
        mock = MockColonyClient(responses={"react_post": _raise_not_found})
        result = self._tool("colony_react_to_post", mock).invoke({"post_id": "p", "emoji": "fire"})
        assert "Error" in result

    def test_react_to_post_async_error(self) -> None:
        mock = MockColonyClient(responses={"react_post": _raise_not_found})
        result = asyncio.run(self._tool("colony_react_to_post", mock).ainvoke({"post_id": "p", "emoji": "fire"}))
        assert "Error" in result

    def test_react_to_comment_error(self) -> None:
        mock = MockColonyClient(responses={"react_comment": _raise_not_found})
        result = self._tool("colony_react_to_comment", mock).invoke({"comment_id": "c", "emoji": "heart"})
        assert "Error" in result

    def test_react_to_comment_async_error(self) -> None:
        mock = MockColonyClient(responses={"react_comment": _raise_not_found})
        result = asyncio.run(self._tool("colony_react_to_comment", mock).ainvoke({"comment_id": "c", "emoji": "heart"}))
        assert "Error" in result

    def test_vote_poll_error(self) -> None:
        mock = MockColonyClient(responses={"vote_poll": _raise_not_found})
        result = self._tool("colony_vote_poll", mock).invoke({"post_id": "p", "option_id": "o"})
        assert "Error" in result

    def test_vote_poll_async_error(self) -> None:
        mock = MockColonyClient(responses={"vote_poll": _raise_not_found})
        result = asyncio.run(self._tool("colony_vote_poll", mock).ainvoke({"post_id": "p", "option_id": "o"}))
        assert "Error" in result

    def test_join_colony_error(self) -> None:
        mock = MockColonyClient(responses={"join_colony": _raise_not_found})
        result = self._tool("colony_join_colony", mock).invoke({"colony": "c"})
        assert "Error" in result

    def test_join_colony_async_error(self) -> None:
        mock = MockColonyClient(responses={"join_colony": _raise_not_found})
        result = asyncio.run(self._tool("colony_join_colony", mock).ainvoke({"colony": "c"}))
        assert "Error" in result

    def test_leave_colony_error(self) -> None:
        mock = MockColonyClient(responses={"leave_colony": _raise_not_found})
        result = self._tool("colony_leave_colony", mock).invoke({"colony": "c"})
        assert "Error" in result

    def test_leave_colony_async_error(self) -> None:
        mock = MockColonyClient(responses={"leave_colony": _raise_not_found})
        result = asyncio.run(self._tool("colony_leave_colony", mock).ainvoke({"colony": "c"}))
        assert "Error" in result

    def test_create_webhook_error(self) -> None:
        mock = MockColonyClient(responses={"create_webhook": _raise_not_found})
        result = self._tool("colony_create_webhook", mock).invoke(
            {"url": "https://x", "events": ["post_created"], "secret": "1234567890123456"}
        )
        assert "Error" in result

    def test_create_webhook_async_error(self) -> None:
        mock = MockColonyClient(responses={"create_webhook": _raise_not_found})
        result = asyncio.run(
            self._tool("colony_create_webhook", mock).ainvoke(
                {"url": "https://x", "events": ["post_created"], "secret": "1234567890123456"}
            )
        )
        assert "Error" in result

    def test_delete_webhook_error(self) -> None:
        mock = MockColonyClient(responses={"delete_webhook": _raise_not_found})
        result = self._tool("colony_delete_webhook", mock).invoke({"webhook_id": "w"})
        assert "Error" in result

    def test_delete_webhook_async_error(self) -> None:
        mock = MockColonyClient(responses={"delete_webhook": _raise_not_found})
        result = asyncio.run(self._tool("colony_delete_webhook", mock).ainvoke({"webhook_id": "w"}))
        assert "Error" in result


# ── callbacks.py uncovered branches ──────────────────────────────────


class TestCallbacksMetadataExtraction:
    """Cover the small `if "x" in inputs` branches in _extract_metadata."""

    def test_extracts_comment_id(self) -> None:
        meta = _extract_metadata("any", {"comment_id": "c1"}, None)
        assert meta["colony.comment_id"] == "c1"

    def test_extracts_user_id(self) -> None:
        meta = _extract_metadata("any", {"user_id": "u1"}, None)
        assert meta["colony.user_id"] == "u1"

    def test_extracts_post_type(self) -> None:
        meta = _extract_metadata("any", {"post_type": "finding"}, None)
        assert meta["colony.post_type"] == "finding"


class TestCallbacksLogging:
    """Cover the optional log_level branches in on_tool_start / end / error."""

    def test_on_tool_start_with_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = ColonyCallbackHandler(log_level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="langchain_colony"):
            handler.on_tool_start({"name": "colony_create_post"}, "{}", run_id="r1", inputs={"title": "t"})
        assert any("colony_create_post" in r.message for r in caplog.records)

    def test_on_tool_end_with_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = ColonyCallbackHandler(log_level=logging.INFO)
        handler.on_tool_start({"name": "colony_get_me"}, "{}", run_id="r1", inputs={})
        with caplog.at_level(logging.INFO, logger="langchain_colony"):
            handler.on_tool_end("ok", run_id="r1")
        assert any("colony_get_me" in r.message for r in caplog.records)

    def test_on_tool_end_unknown_run_id(self) -> None:
        """on_tool_end with no matching pending action should be a no-op."""
        handler = ColonyCallbackHandler()
        handler.on_tool_end("anything", run_id="never-started")
        assert handler.actions == []

    def test_on_tool_error_unknown_run_id(self) -> None:
        """on_tool_error with no matching pending action should be a no-op."""
        handler = ColonyCallbackHandler()
        handler.on_tool_error(RuntimeError("x"), run_id="never-started")
        assert handler.actions == []

    def test_on_tool_error_with_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = ColonyCallbackHandler(log_level=logging.INFO)
        handler.on_tool_start({"name": "colony_get_post"}, "{}", run_id="r1", inputs={"post_id": "p"})
        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            handler.on_tool_error(RuntimeError("kaboom"), run_id="r1")
        assert any("FAILED" in r.message for r in caplog.records)


# ── events.py uncovered branches ─────────────────────────────────────


class TestEventPollerErrorPaths:
    """Cover the error/exception branches in ColonyEventPoller."""

    def test_mark_read_failure_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """mark_notifications_read failures during polling are logged but don't crash."""
        call_count = {"n": 0}

        def get_notifs(**_kw: Any) -> dict:
            call_count["n"] += 1
            return {"notifications": [{"id": "n1", "type": "reply", "actor": {"username": "x"}}]}

        def boom() -> None:
            raise ColonyAPIError("mark failed", status=500)

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        mock.mark_notifications_read = boom  # type: ignore[method-assign]
        poller = ColonyEventPoller(client=mock, mark_read=True)

        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            poller.poll_once()

        assert any("Failed to mark notifications read" in r.message for r in caplog.records)

    def test_poll_once_async_failure_returns_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        """poll_once_async swallows exceptions and returns []."""
        mock = MockColonyClient(responses={"get_notifications": _raise_generic})
        poller = ColonyEventPoller(client=mock)

        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            result = asyncio.run(poller.poll_once_async())

        assert result == []
        assert any("Failed to poll notifications" in r.message for r in caplog.records)

    def test_poll_once_async_dispatches(self) -> None:
        """poll_once_async dispatches to handlers (sync + async)."""
        seen: list[str] = []

        def sync_handler(notif: Any) -> None:
            seen.append(f"sync:{notif.id}")

        async def async_handler(notif: Any) -> None:
            seen.append(f"async:{notif.id}")

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "n42", "type": "reply", "actor": {"username": "x"}}]}

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        poller = ColonyEventPoller(client=mock)
        poller.add_handler(sync_handler, "reply")
        poller.add_handler(async_handler, None)  # catch-all

        asyncio.run(poller.poll_once_async())

        assert "sync:n42" in seen
        assert "async:n42" in seen

    def test_poll_once_async_mark_read_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """async mark_notifications_read failures are logged."""

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "nx", "type": "reply", "actor": {"username": "x"}}]}

        def boom() -> None:
            raise ColonyAPIError("nope", status=500)

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        mock.mark_notifications_read = boom  # type: ignore[method-assign]
        poller = ColonyEventPoller(client=mock, mark_read=True)

        with caplog.at_level(logging.WARNING, logger="langchain_colony"):
            asyncio.run(poller.poll_once_async())

        assert any("Failed to mark notifications read" in r.message for r in caplog.records)

    def test_run_async_stops_on_flag(self) -> None:
        """run_async loops until _async_stop is set."""
        mock = MockColonyClient(responses={"get_notifications": {"notifications": []}})
        poller = ColonyEventPoller(client=mock)

        async def go() -> None:
            task = asyncio.create_task(poller.run_async(poll_interval=0.01))
            await asyncio.sleep(0.03)
            poller._async_stop = True  # type: ignore[attr-defined]
            await task

        asyncio.run(go())  # should return cleanly

    def test_dispatch_handler_exception_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A handler that raises is caught and logged, doesn't break polling."""

        def boom_handler(_notif: Any) -> None:
            raise RuntimeError("handler exploded")

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "ne", "type": "reply", "actor": {"username": "x"}}]}

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        poller = ColonyEventPoller(client=mock)
        poller.add_handler(boom_handler, "reply")

        with caplog.at_level(logging.ERROR, logger="langchain_colony"):
            poller.poll_once()

        assert any("handler exploded" in r.message or "Handler error" in r.message for r in caplog.records)

    def test_dispatch_catchall_handler_exception_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom_handler(_notif: Any) -> None:
            raise RuntimeError("catchall exploded")

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "n_ca", "type": "mention", "actor": {"username": "x"}}]}

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        poller = ColonyEventPoller(client=mock)
        poller.add_handler(boom_handler, None)

        with caplog.at_level(logging.ERROR, logger="langchain_colony"):
            poller.poll_once()

        assert any("catch-all" in r.message for r in caplog.records)

    def test_dispatch_async_handler_exception_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        async def boom_handler(_notif: Any) -> None:
            raise RuntimeError("async handler boom")

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "n_ah", "type": "reply", "actor": {"username": "x"}}]}

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        poller = ColonyEventPoller(client=mock)
        poller.add_handler(boom_handler, "reply")

        with caplog.at_level(logging.ERROR, logger="langchain_colony"):
            asyncio.run(poller.poll_once_async())

        assert any("async handler boom" in r.message or "Handler error" in r.message for r in caplog.records)

    def test_dispatch_async_catchall_exception_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        async def boom_handler(_notif: Any) -> None:
            raise RuntimeError("async catchall boom")

        def get_notifs(**_kw: Any) -> dict:
            return {"notifications": [{"id": "n_ac", "type": "dm", "actor": {"username": "x"}}]}

        mock = MockColonyClient(responses={"get_notifications": get_notifs})
        poller = ColonyEventPoller(client=mock)
        poller.add_handler(boom_handler, None)

        with caplog.at_level(logging.ERROR, logger="langchain_colony"):
            asyncio.run(poller.poll_once_async())

        assert any("catch-all" in r.message for r in caplog.records)


# ── retriever.py uncovered branches ──────────────────────────────────


class TestAsyncHappyPaths:
    """Cover the success-path return statements in async tools.

    The base test_toolkit.py covers the sync happy paths, but several
    tools' `_arun` happy-path returns aren't reached without explicit
    async invocation.
    """

    def _tool(self, name: str, mock: MockColonyClient) -> Any:
        toolkit = ColonyToolkit(client=mock)
        return next(t for t in toolkit.get_tools() if t.name == name)

    def test_async_unfollow_user(self) -> None:
        mock = MockColonyClient(responses={"unfollow": {"following": False}})
        result = asyncio.run(self._tool("colony_unfollow_user", mock).ainvoke({"user_id": "u1"}))
        assert "Unfollowed" in result or "u1" in result

    def test_async_react_to_comment(self) -> None:
        mock = MockColonyClient(responses={"react_comment": {"toggled": True}})
        result = asyncio.run(self._tool("colony_react_to_comment", mock).ainvoke({"comment_id": "c1", "emoji": "fire"}))
        assert "c1" in result

    def test_async_get_poll(self) -> None:
        mock = MockColonyClient(
            responses={"get_poll": {"options": [{"id": "o", "text": "Yes", "votes": 3}], "total_votes": 3}}
        )
        result = asyncio.run(self._tool("colony_get_poll", mock).ainvoke({"post_id": "p1"}))
        assert "Poll" in result or "Yes" in result

    def test_async_leave_colony(self) -> None:
        mock = MockColonyClient(responses={"leave_colony": {"left": True}})
        result = asyncio.run(self._tool("colony_leave_colony", mock).ainvoke({"colony": "general"}))
        assert "Left" in result or "general" in result

    def test_async_get_webhooks(self) -> None:
        mock = MockColonyClient(responses={"get_webhooks": {"webhooks": []}})
        result = asyncio.run(self._tool("colony_get_webhooks", mock).ainvoke({}))
        assert "No webhooks" in result

    def test_async_delete_webhook(self) -> None:
        mock = MockColonyClient(responses={"delete_webhook": {"success": True}})
        result = asyncio.run(self._tool("colony_delete_webhook", mock).ainvoke({"webhook_id": "w1"}))
        assert "Deleted" in result or "w1" in result

    def test_async_update_profile_no_fields(self) -> None:
        """Async update_profile with no fields short-circuits to a friendly string."""
        mock = MockColonyClient()
        result = asyncio.run(self._tool("colony_update_profile", mock).ainvoke({}))
        assert "No fields" in result


# ── tools.py format helper edge cases ────────────────────────────────


class TestFormatHelpersEdgeCases:
    def test_format_poll_with_non_dict(self) -> None:
        from langchain_colony.tools import _format_poll

        # Non-dict input falls through to str(data).
        assert _format_poll("an error string") == "an error string"

    def test_format_webhooks_with_non_dict_non_list(self) -> None:
        from langchain_colony.tools import _format_webhooks

        # Non-dict, non-list input falls through to str(data).
        assert _format_webhooks("an error string") == "an error string"


# ── retriever.py async path ──────────────────────────────────────────


class TestRetrieverErrorPath:
    """Cover the `except Exception: pass` branches in both
    sync and async _enrich_with_comments paths.
    """

    def test_enrich_swallows_get_post_error(self) -> None:
        """When get_post raises during enrichment, the doc still comes back unmodified."""

        def get_post_boom(**_kw: Any) -> dict:
            raise RuntimeError("post fetch failed")

        # The retriever uses iter_posts; MockColonyClient.iter_posts yields
        # whatever's in get_posts response under either "items" or "posts".
        mock = MockColonyClient(
            responses={
                "get_posts": {
                    "items": [
                        {
                            "id": "p1",
                            "title": "T",
                            "body": "B",
                            "post_type": "discussion",
                            "score": 1,
                            "comment_count": 1,
                            "author": {"username": "a"},
                            "colony": {"name": "general"},
                        }
                    ]
                },
                "get_post": get_post_boom,
            }
        )
        retriever = ColonyRetriever(client=mock, include_comments=True)
        docs = retriever.invoke("test query")
        assert len(docs) == 1
        # Body is the original snippet — comment-enrichment failure was swallowed.
        assert "B" in docs[0].page_content

    def test_async_enrich_swallows_get_post_error(self) -> None:
        """The async enrichment path also swallows get_post failures."""

        def get_post_boom(**_kw: Any) -> dict:
            raise RuntimeError("post fetch failed")

        mock = MockColonyClient(
            responses={
                "get_posts": {
                    "items": [
                        {
                            "id": "p2",
                            "title": "T2",
                            "body": "B2",
                            "post_type": "discussion",
                            "score": 1,
                            "comment_count": 1,
                            "author": {"username": "a"},
                            "colony": {"name": "general"},
                        }
                    ]
                },
                "get_post": get_post_boom,
            }
        )
        retriever = ColonyRetriever(client=mock, include_comments=True)
        docs = asyncio.run(retriever.ainvoke("test query"))
        assert len(docs) == 1
        assert "B2" in docs[0].page_content


# ── __init__.py lazy import ──────────────────────────────────────────


class TestLazyImports:
    def test_create_colony_agent_lazy_import(self) -> None:
        """Accessing create_colony_agent through the package triggers __getattr__.

        Skipped when ``langgraph`` isn't installed (the lazy import imports
        ``langchain_colony.agent``, which imports langgraph at module top).
        CI installs langgraph as a dev dep so this should always run there.
        """
        pytest.importorskip("langgraph", reason="langgraph not installed")
        import langchain_colony
        from langchain_colony.agent import create_colony_agent

        attr = langchain_colony.create_colony_agent
        assert attr is create_colony_agent

    def test_unknown_attribute_raises(self) -> None:
        import langchain_colony

        with pytest.raises(AttributeError, match="no attribute 'definitely_not_a_thing'"):
            _ = langchain_colony.definitely_not_a_thing  # type: ignore[attr-defined]


# ── New batch tools (v0.7.0) ─────────────────────────────────────────


class TestBatchTools:
    """Cover the new colony_get_posts_by_ids / colony_get_users_by_ids tools."""

    def _tool(self, name: str, mock: MockColonyClient) -> Any:
        toolkit = ColonyToolkit(client=mock)
        return next(t for t in toolkit.get_tools() if t.name == name)

    def test_get_posts_by_ids_returns_posts(self) -> None:
        def stub(post_ids: list[str]) -> list:
            return [
                {
                    "id": pid,
                    "title": f"Post {pid}",
                    "post_type": "discussion",
                    "score": 1,
                    "comment_count": 0,
                    "author": {"username": "a"},
                    "colony": {"name": "general"},
                }
                for pid in post_ids
            ]

        mock = MockColonyClient()
        mock.get_posts_by_ids = stub  # type: ignore[method-assign]
        result = self._tool("colony_get_posts_by_ids", mock).invoke({"post_ids": ["p1", "p2"]})
        assert "Post p1" in result
        assert "Post p2" in result

    def test_get_posts_by_ids_async_returns_posts(self) -> None:
        async def stub(post_ids: list[str]) -> list:
            return [
                {
                    "id": pid,
                    "title": f"Post {pid}",
                    "post_type": "discussion",
                    "score": 1,
                    "comment_count": 0,
                    "author": {"username": "a"},
                    "colony": {"name": "general"},
                }
                for pid in post_ids
            ]

        mock = MockColonyClient()
        mock.get_posts_by_ids = stub  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_posts_by_ids", mock).ainvoke({"post_ids": ["p1"]}))
        assert "Post p1" in result

    def test_get_posts_by_ids_empty(self) -> None:
        def stub(post_ids: list[str]) -> list:
            return []

        mock = MockColonyClient()
        mock.get_posts_by_ids = stub  # type: ignore[method-assign]
        result = self._tool("colony_get_posts_by_ids", mock).invoke({"post_ids": ["bogus"]})
        assert "No posts found" in result

    def test_get_posts_by_ids_async_empty(self) -> None:
        async def stub(post_ids: list[str]) -> list:
            return []

        mock = MockColonyClient()
        mock.get_posts_by_ids = stub  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_posts_by_ids", mock).ainvoke({"post_ids": ["bogus"]}))
        assert "No posts found" in result

    def test_get_posts_by_ids_error(self) -> None:
        def boom(post_ids: list[str]) -> list:
            raise ColonyNotFoundError("nope", status=404)

        mock = MockColonyClient()
        mock.get_posts_by_ids = boom  # type: ignore[method-assign]
        result = self._tool("colony_get_posts_by_ids", mock).invoke({"post_ids": ["x"]})
        assert "Error" in result

    def test_get_posts_by_ids_async_error(self) -> None:
        async def boom(post_ids: list[str]) -> list:
            raise ColonyNotFoundError("nope", status=404)

        mock = MockColonyClient()
        mock.get_posts_by_ids = boom  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_posts_by_ids", mock).ainvoke({"post_ids": ["x"]}))
        assert "Error" in result

    def test_get_users_by_ids_returns_users(self) -> None:
        def stub(user_ids: list[str]) -> list:
            return [{"username": uid, "display_name": uid.upper()} for uid in user_ids]

        mock = MockColonyClient()
        mock.get_users_by_ids = stub  # type: ignore[method-assign]
        result = self._tool("colony_get_users_by_ids", mock).invoke({"user_ids": ["alice", "bob"]})
        assert "alice" in result
        assert "bob" in result

    def test_get_users_by_ids_async_returns_users(self) -> None:
        async def stub(user_ids: list[str]) -> list:
            return [{"username": uid, "display_name": uid.upper()} for uid in user_ids]

        mock = MockColonyClient()
        mock.get_users_by_ids = stub  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_users_by_ids", mock).ainvoke({"user_ids": ["alice"]}))
        assert "alice" in result

    def test_get_users_by_ids_empty(self) -> None:
        def stub(user_ids: list[str]) -> list:
            return []

        mock = MockColonyClient()
        mock.get_users_by_ids = stub  # type: ignore[method-assign]
        result = self._tool("colony_get_users_by_ids", mock).invoke({"user_ids": ["bogus"]})
        assert "No users found" in result

    def test_get_users_by_ids_async_empty(self) -> None:
        async def stub(user_ids: list[str]) -> list:
            return []

        mock = MockColonyClient()
        mock.get_users_by_ids = stub  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_users_by_ids", mock).ainvoke({"user_ids": ["bogus"]}))
        assert "No users found" in result

    def test_get_users_by_ids_error(self) -> None:
        def boom(user_ids: list[str]) -> list:
            raise ColonyNotFoundError("nope", status=404)

        mock = MockColonyClient()
        mock.get_users_by_ids = boom  # type: ignore[method-assign]
        result = self._tool("colony_get_users_by_ids", mock).invoke({"user_ids": ["x"]})
        assert "Error" in result

    def test_get_users_by_ids_async_error(self) -> None:
        async def boom(user_ids: list[str]) -> list:
            raise ColonyNotFoundError("nope", status=404)

        mock = MockColonyClient()
        mock.get_users_by_ids = boom  # type: ignore[method-assign]
        result = asyncio.run(self._tool("colony_get_users_by_ids", mock).ainvoke({"user_ids": ["x"]}))
        assert "Error" in result


# ── typed=True passthrough (v0.7.0) ──────────────────────────────────


class TestTypedPassthrough:
    """ColonyToolkit / AsyncColonyToolkit forward typed=True to the SDK client."""

    def test_sync_toolkit_typed_passthrough(self) -> None:
        from colony_sdk import ColonyClient

        toolkit = ColonyToolkit(api_key="col_test", typed=True)
        assert isinstance(toolkit.client, ColonyClient)
        assert toolkit.client.typed is True

    def test_sync_toolkit_typed_default_false(self) -> None:
        from colony_sdk import ColonyClient

        toolkit = ColonyToolkit(api_key="col_test")
        assert isinstance(toolkit.client, ColonyClient)
        assert toolkit.client.typed is False

    def test_async_toolkit_typed_passthrough(self) -> None:
        from langchain_colony import AsyncColonyToolkit

        toolkit = AsyncColonyToolkit(api_key="col_test", typed=True)
        assert toolkit.client.typed is True

    def test_async_toolkit_typed_default_false(self) -> None:
        from langchain_colony import AsyncColonyToolkit

        toolkit = AsyncColonyToolkit(api_key="col_test")
        assert toolkit.client.typed is False
