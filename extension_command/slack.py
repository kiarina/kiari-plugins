# Usage:
#   kiari ext -v slack post-message --channel C01234567 "hello"
#
#   The examples below omit the `kiari ext -v slack` prefix:
#     post-message --channel C01234567 --thread-ts 1700000000.000100 "reply"
#     get-channel-messages --channel C01234567 --limit 20
#     watch-channel
#     watch-channel --channel C01234567,C07654321
import argparse
import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import kiarina.lib.slack
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Main
# --------------------------------------------------


class SlackCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)

        if options.command == "post-message":
            await self._post_message(options)
        elif options.command == "get-channel-messages":
            await self._get_channel_messages(options)
        elif options.command == "watch-channel":
            await self._watch_channel(options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    # ----- post-message -----

    async def _post_message(self, options: argparse.Namespace) -> None:
        client = AsyncWebClient(token=self._bot_token(options))

        response = await client.chat_postMessage(
            channel=options.channel,
            text=options.message,
            thread_ts=options.thread_ts if options.thread_ts else None,
        )

        print("Posted message:")
        print(f"  channel: {response['channel']}")
        print(f"  ts: {response['ts']}")
        print(f"  text: {options.message}")

    # ----- get-channel-messages -----

    async def _get_channel_messages(self, options: argparse.Namespace) -> None:
        client = AsyncWebClient(token=self._bot_token(options))

        request: dict[str, Any] = {
            "channel": options.channel,
            "limit": options.limit,
        }
        if options.oldest:
            request["oldest"] = options.oldest
        if options.latest:
            request["latest"] = options.latest

        response = await client.conversations_history(**request)
        messages: list[dict[str, Any]] = response.get("messages", []) or []

        print(f"Fetched {len(messages)} messages from {options.channel}")

        for message in messages:
            print("-" * 20)
            _print_message(message)

    # ----- watch-channel -----

    async def _watch_channel(self, options: argparse.Namespace) -> None:
        bot_token = self._bot_token(options)
        app_token = self._app_token(options)

        channel_filter: set[str] = set(options.channel or [])

        app = AsyncApp(token=bot_token)

        @app.event("message")
        async def _on_message(event: dict[str, Any], client: AsyncWebClient) -> None:
            print("Received message event:")
            try:
                if event.get("bot_id"):
                    return

                subtype = event.get("subtype")
                if subtype in ("message_changed", "message_deleted", "channel_join"):
                    return

                channel = event.get("channel", "")
                if channel_filter and channel not in channel_filter:
                    return

                print("-" * 20)
                _print_message(event)

            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)

        @app.event("app_mention")
        async def _on_app_mention(event: dict[str, Any]) -> None:
            pass

        handler = AsyncSocketModeHandler(app, app_token)

        if channel_filter:
            print(f"Watching channels: {sorted(channel_filter)}")
        else:
            print("Watching all channels the bot is in.")
        print("Listening. Press Ctrl+C to stop.")

        handler_task = asyncio.create_task(handler.start_async())  # type: ignore[no-untyped-call]

        try:
            await handler_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopping.")
        finally:
            await handler.close_async()  # type: ignore[no-untyped-call]
            if not handler_task.done():
                handler_task.cancel()
                try:
                    await handler_task
                except (asyncio.CancelledError, Exception):
                    pass

        print("Stopped.")

    # ----- tokens -----

    def _slack_settings(self, options: argparse.Namespace) -> kiarina.lib.slack.SlackSettings:
        return kiarina.lib.slack.settings_manager.get_settings(options.slack_settings_key)

    def _bot_token(self, options: argparse.Namespace) -> str:
        settings = self._slack_settings(options)
        if not settings.bot_token:
            raise ValueError("Slack Bot Token is not configured")
        return settings.bot_token.get_secret_value()

    def _app_token(self, options: argparse.Namespace) -> str:
        settings = self._slack_settings(options)
        if not settings.app_token:
            raise ValueError("Slack App Token is not configured")
        return settings.app_token.get_secret_value()


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _print_message(message: dict[str, Any]) -> None:
    ts = message.get("ts", "")
    print(f"ts: {ts}")

    if ts:
        try:
            dt = datetime.fromtimestamp(float(ts), tz=UTC)
            print(f"time: {dt.isoformat()}")
        except (ValueError, TypeError):
            pass

    if user := message.get("user"):
        print(f"user: {user}")
    if channel := message.get("channel"):
        print(f"channel: {channel}")
    if team := message.get("team"):
        print(f"team: {team}")
    if thread_ts := message.get("thread_ts"):
        print(f"thread_ts: {thread_ts}")
    if subtype := message.get("subtype"):
        print(f"subtype: {subtype}")

    text = message.get("text", "")
    print(f"text: {text}")

    files = message.get("files") or []
    for file_data in files:
        name = file_data.get("name", "")
        size = file_data.get("size", 0)
        print(f"file: {name} ({size} bytes)")


def _parse_channels(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Post / fetch / watch Slack messages (single-workspace).",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    # fmt: off
    common_parser.add_argument("--slack-settings-key", default=None, help="Settings key passed to kiarina.lib.slack.settings_manager.get_settings.")
    # fmt: on

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- post-message -----
    post_parser = subparsers.add_parser(
        "post-message", parents=[common_parser], help="Post a message to a channel."
    )
    # fmt: off
    post_parser.add_argument("--channel", required=True, help="Channel ID (e.g. C01234567).")
    post_parser.add_argument("--thread-ts", default=None, help="Reply in a thread by ts.")
    post_parser.add_argument("message", help="Message text (UTF-8).")
    # fmt: on

    # ----- get-channel-messages -----
    get_parser = subparsers.add_parser(
        "get-channel-messages",
        parents=[common_parser],
        help="Fetch recent messages from a channel.",
    )
    # fmt: off
    get_parser.add_argument("--channel", required=True, help="Channel ID.")
    get_parser.add_argument("--limit", type=int, default=20, help="Max number of messages.")
    get_parser.add_argument("--oldest", default=None, help="Oldest message ts (inclusive).")
    get_parser.add_argument("--latest", default=None, help="Latest message ts (inclusive).")
    # fmt: on

    # ----- watch-channel -----
    watch_parser = subparsers.add_parser(
        "watch-channel",
        parents=[common_parser],
        help="Watch new messages via Socket Mode (requires app_token).",
    )
    # fmt: off
    watch_parser.add_argument("--channel", type=_parse_channels, default=None, help="Comma-separated channel IDs to filter. Default: all channels the bot is in.")
    # fmt: on

    return parser.parse_args(list(args))


extension_command_registry.register("slack", SlackCommand)
