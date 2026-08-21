# Usage:
#   kiari ext -v firebase-rtdb get --database-url https://my-project.firebaseio.com --path /watch/test
#
#   Authentication is delegated to settings: kiarina.lib.firebase supplies api_key and
#   token_file_path, and kiarina.lib.firebase_rtdb.firebase_settings_key selects which
#   of them to use. Seed the token set with `kiari ext firebase login`, which writes it to
#   the same token_file_path.
#
#   The examples below omit the `kiari ext -v firebase-rtdb` prefix:
#     get --database-url https://my-project.firebaseio.com --path /watch/test --output-file ./.tmp/rtdb/get.json
#     set --database-url https://my-project.firebaseio.com --path /watch/test '{"message": "hello"}'
#     set --database-url https://my-project.firebaseio.com --path /watch/test --patch '{"extra": "field"}'
#     set --database-url https://my-project.firebaseio.com --path /watch/test --from-file ./payload.json
#     set --database-url https://my-project.firebaseio.com --path /watch/test null    # delete
#     watch --database-url https://my-project.firebaseio.com --path /watch/test
import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from kiarina.lib.firebase import token_manager_registry
from kiarina.lib.firebase_rtdb import get_data, watch_data

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _build_url(database_url: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{database_url.rstrip('/')}{path}.json"


def _load_value(options: argparse.Namespace) -> Any:
    if options.from_file:
        text = Path(options.from_file).expanduser().read_text(encoding="utf-8")
    elif options.value is not None:
        text = options.value
    else:
        raise ValueError("Either a positional value or --from-file is required.")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse value as JSON: {e}. "
            "Pass a JSON literal (e.g. '\"hello\"', '123', 'null', '{\"k\":1}'), "
            "or use --from-file to load from a JSON file."
        ) from e


# --------------------------------------------------
# Main
# --------------------------------------------------


class FirebaseRTDBCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)

        if options.command == "set":
            await self._set(options)
        elif options.command == "get":
            await self._get(options)
        elif options.command == "watch":
            await self._watch(options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    # ----- set -----

    async def _set(self, options: argparse.Namespace) -> None:
        value = _load_value(options)
        token = await token_manager_registry.get().get_token()

        url = _build_url(options.database_url, options.path)
        method = "PATCH" if options.patch else "PUT"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.request(
                method,
                url,
                params={"auth": token.id_token},
                json=value,
            )
            response.raise_for_status()
            result = response.json() if response.content else None

        print(f"{method} {options.database_url}{options.path}")
        print(f"  value: {json.dumps(value, ensure_ascii=False)}")
        print(f"  response: {json.dumps(result, ensure_ascii=False)}")

    # ----- get -----

    async def _get(self, options: argparse.Namespace) -> None:
        data = await get_data(
            database_url=options.database_url,
            path=options.path,
        )

        text = json.dumps(data, ensure_ascii=False, indent=2)
        print(f"GET {options.database_url}{options.path}")
        print("-" * 20)
        print(text)
        print("-" * 20)

        if options.output_file:
            output_path = Path(options.output_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
            print(f"Saved to: {output_path}")

    # ----- watch -----

    async def _watch(self, options: argparse.Namespace) -> None:
        stop_event = asyncio.Event()

        print(f"WATCH {options.database_url}{options.path}")
        print("Listening. Press Ctrl+C to stop.")
        print("-" * 20)

        received_count = 0

        try:
            async for event in watch_data(
                database_url=options.database_url,
                path=options.path,
                stop_event=stop_event,
            ):
                received_count += 1
                print(f"event_type: {event.event_type}")
                print(f"path: {event.path}")
                print(f"data: {json.dumps(event.data, ensure_ascii=False)}")
                print("-" * 20)

                if options.limit is not None and received_count >= options.limit:
                    stop_event.set()
                    break

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopped.")
            stop_event.set()

        print(f"Total received: {received_count}")


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Read, write, and watch Firebase Realtime Database data.",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    # fmt: off
    common_parser.add_argument("--database-url", required=True, help="Firebase database URL (e.g. https://my-project.firebaseio.com).")
    common_parser.add_argument("--path", default="/", help="Database path (default: /).")
    # fmt: on

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- set -----
    set_parser = subparsers.add_parser(
        "set",
        parents=[common_parser],
        help="Write data to the given path (PUT, or PATCH with --patch).",
    )
    # fmt: off
    set_parser.add_argument("value", nargs="?", default=None, help="JSON literal to write (e.g. '\"hello\"', '123', 'null', '{\"k\":1}'). Use null to delete.")
    set_parser.add_argument("--from-file", default=None, help="Read JSON value from this file instead of the positional argument.")
    set_parser.add_argument("--patch", action="store_true", help="Use PATCH (merge update) instead of PUT (overwrite).")
    # fmt: on

    # ----- get -----
    get_parser = subparsers.add_parser(
        "get",
        parents=[common_parser],
        help="Read data at the given path.",
    )
    get_parser.add_argument(
        "--output-file",
        default=None,
        help="Write the fetched JSON to this file (in addition to stdout).",
    )

    # ----- watch -----
    watch_parser = subparsers.add_parser(
        "watch",
        parents=[common_parser],
        help="Stream data change events at the given path until stopped.",
    )
    watch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after receiving this many events. Default: no limit.",
    )

    return parser.parse_args(list(args))


extension_command_registry.register("firebase-rtdb", FirebaseRTDBCommand)
