# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/firebase.py" firebase login --uid kiarina
#
#   login mints a custom token from the configured Google service account, exchanges it for
#   a Firebase token set, and writes it where token_manager_registry reads from. Every other
#   command that talks to Firebase picks it up from there.
#
#   The examples below omit the `kiari ext -v --plugin ... firebase` prefix:
#     login --uid kiarina --token-data-file-path ./.tmp/firebase/token.json
#     login --uid kiarina --firebase-settings-key staging
import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from kiarina.lib.firebase import (
    FileTokenStore,
    exchange_custom_token,
    settings_manager as firebase_auth_settings_manager,
)
from kiarina.lib.google import settings_manager as google_auth_settings_manager

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _create_custom_token(options: argparse.Namespace) -> str:
    try:
        import firebase_admin  # type: ignore[import-untyped]
        from firebase_admin import (
            auth,
            credentials,
            initialize_app,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "firebase_admin is required to log in. Install the firebase-admin package."
        ) from e

    google_auth_settings = google_auth_settings_manager.get_settings(
        options.google_auth_settings_key
    )

    if google_auth_settings.service_account_file:
        credential = credentials.Certificate(google_auth_settings.service_account_file)
    elif service_account_data := google_auth_settings.get_service_account_data():
        credential = credentials.Certificate(service_account_data)
    else:
        raise ValueError(
            "Google service account is not configured. Set service_account_file "
            "or service_account_data in kiarina.lib.google settings."
        )

    try:
        app = firebase_admin.get_app()
    except ValueError:
        app = initialize_app(credential)

    custom_token: bytes = auth.create_custom_token(options.uid, app=app)
    return custom_token.decode("utf-8")


# --------------------------------------------------
# Main
# --------------------------------------------------


class FirebaseCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)

        if options.command == "login":
            await self._login(options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    # ----- login -----

    async def _login(self, options: argparse.Namespace) -> None:
        firebase_auth_settings = firebase_auth_settings_manager.get_settings(
            options.firebase_settings_key
        )
        custom_token = await asyncio.to_thread(_create_custom_token, options)
        token_data = await exchange_custom_token(
            custom_token,
            firebase_auth_settings.api_key.get_secret_value(),
        )

        token_data_file_path = (
            options.token_data_file_path or firebase_auth_settings.token_data_file_path
        )

        if not token_data_file_path:
            raise ValueError(
                "No output path. Pass --token-data-file-path, or set token_data_file_path "
                "in the kiarina.lib.firebase settings that token_manager_registry reads."
            )

        output_path = Path(token_data_file_path).expanduser()
        await FileTokenStore(str(output_path)).set(token_data)

        print(f"Logged in as: {options.uid}")
        print(f"  token data: {output_path}")
        print(f"  expires_at: {token_data.expires_at.isoformat()}")


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Authenticate against Firebase.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- login -----
    login_parser = subparsers.add_parser(
        "login",
        help="Mint a Firebase token set from a service account and store it.",
    )
    # fmt: off
    login_parser.add_argument("--uid", default="kiarina", help="Firebase custom token UID.")
    login_parser.add_argument("--token-data-file-path", default=None, help="Where to store the token set. Defaults to token_data_file_path in the kiarina.lib.firebase settings.")
    login_parser.add_argument("--firebase-settings-key", default=None, help="Settings key passed to kiarina.lib.firebase.settings_manager.get_settings.")
    login_parser.add_argument("--google-auth-settings-key", default=None, help="Settings key passed to kiarina.lib.google.settings_manager.get_settings.")
    # fmt: on

    return parser.parse_args(list(args))


extension_command_registry.register("firebase", FirebaseCommand)
