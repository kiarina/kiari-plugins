# Usage:
#   kiari ext -v google --google-settings-key hoge
#
#   The examples below omit the `kiari ext -v google` prefix:
#     --google-settings-key hoge --port 8080
#     --scope https://www.googleapis.com/auth/cloud-platform --scope https://www.googleapis.com/auth/drive --scope https://www.googleapis.com/auth/spreadsheets
import argparse
from collections.abc import Sequence
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from kiarina.lib.google import GoogleSettings, settings_manager

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)


class GoogleCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)
        settings = settings_manager.get_settings(options.google_settings_key)
        scopes = options.scopes if options.scopes is not None else settings.scopes

        _validate_settings(settings, scopes=scopes)
        flow = _create_flow(settings, scopes=scopes)
        credentials = flow.run_local_server(port=options.port)
        credentials_json = credentials.to_json()

        print("Authentication successful.")
        print(credentials_json)

        if settings.authorized_user_file:
            output_path = Path(settings.authorized_user_file).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(credentials_json, encoding="utf-8")
            print(f"Credentials saved to: {output_path}")


def _validate_settings(settings: GoogleSettings, *, scopes: list[str]) -> None:
    if settings.type != "user_account":
        raise ValueError(
            f"Google settings must use type 'user_account', but got {settings.type!r}."
        )

    if not scopes:
        raise ValueError("Google user account scopes are required.")

    if not settings.client_secret_data and not settings.client_secret_file:
        raise ValueError(
            "Google user account settings require client_secret_data or client_secret_file."
        )


def _create_flow(
    settings: GoogleSettings,
    *,
    scopes: list[str],
) -> InstalledAppFlow:
    if client_secret_data := settings.get_client_secret_data():
        return InstalledAppFlow.from_client_config(
            client_secret_data,
            scopes=scopes,
        )

    if settings.client_secret_file:
        return InstalledAppFlow.from_client_secrets_file(
            settings.client_secret_file,
            scopes=scopes,
        )

    raise AssertionError("Google user account client secret not set")


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Authenticate a Google user account and output its credentials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  kiari ext google --google-settings-key hoge
  kiari ext google \\
    --scope https://www.googleapis.com/auth/cloud-platform \\
    --scope https://www.googleapis.com/auth/drive \\
    --scope https://www.googleapis.com/auth/spreadsheets""",
    )
    # fmt: off
    parser.add_argument("--google-settings-key", default=None, help="kiarina-lib-google settings key. Uses the default key when omitted.")
    parser.add_argument("--scope", dest="scopes", action="append", default=None, help="OAuth scope. Repeatable. Overrides configured scopes when provided.")
    parser.add_argument("--port", type=int, default=8080, help="Local OAuth callback server port. Default: 8080.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("google", GoogleCommand)
