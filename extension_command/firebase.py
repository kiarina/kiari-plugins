# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/firebase.py" firebase login --project-id my-project --uid kiarina
#
#   login mints a custom token with the credential that kiarina.lib.google resolves, exchanges
#   it for a Firebase token set, and writes it to token_file_path in the kiarina.lib.firebase
#   settings -- the same setting token_manager_registry reads from, so every other command
#   that talks to Firebase picks it up. Set it, or login fails.
#
#   How the token gets signed depends on which credential kiarina.lib.google resolves:
#     service account key  ->  signed locally with its private key
#     impersonation        ->  signed by the target principal through the IAM API
#     GCE / Cloud Run      ->  signer discovered from the metadata server
#     user account         ->  nothing to sign with, so --service-account-id is required
#   Application default credentials resolve to one of the first three, depending on the host.
#
#   --service-account-id does not impersonate: the credential still authenticates as itself and
#   only borrows the named service account's key to sign one blob. It needs signBlob permission
#   on that account, the same grant kiarina.lib.google's impersonate_service_account requires --
#   configure that instead and the flag becomes unnecessary. Either way the minted token is
#   issued as (iss / sub) that service account, because Firebase only trusts service account keys.
#
#   --project-id and --uid are required: they say what the token is minted as, and the
#   settings have nowhere to hold them. Where it lands is not an argument, so that login
#   and the readers cannot disagree about the path.
#
#   The examples below omit the `kiari ext -v --plugin ... firebase` prefix:
#     login --project-id my-project --uid kiarina
#     login --project-id my-project --uid kiarina --firebase-settings-key staging
#     login --project-id my-project --uid kiarina --google-auth-settings-key user --service-account-id sa@my-project.iam.gserviceaccount.com
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
from kiarina.lib.google import Credentials, get_credentials

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _create_custom_token(options: argparse.Namespace, *, project_id: str) -> str:
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

    google_credentials = get_credentials(options.google_auth_settings_key)

    # firebase_admin is an optional import, so its base class is only reachable from here.
    class _ResolvedCredential(credentials.Base):  # type: ignore[misc]
        def get_credential(self) -> Credentials:
            return google_credentials

    app_options = {"projectId": project_id}

    if options.service_account_id:
        app_options["serviceAccountId"] = options.service_account_id

    try:
        app = firebase_admin.get_app()
    except ValueError:
        app = initialize_app(_ResolvedCredential(), app_options)

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

        if not firebase_auth_settings.token_file_path:
            raise ValueError(
                "No output path. Set token_file_path in the kiarina.lib.firebase settings "
                "(KIARINA_LIB_FIREBASE_TOKEN_FILE_PATH); token_manager_registry reads the "
                "token set back from there."
            )

        output_path = Path(firebase_auth_settings.token_file_path).expanduser()

        custom_token = await asyncio.to_thread(
            _create_custom_token, options, project_id=options.project_id
        )
        token = await exchange_custom_token(
            custom_token,
            firebase_auth_settings.api_key.get_secret_value(),
        )

        await FileTokenStore(str(output_path)).set(token)

        print(f"Logged in as: {options.uid}")
        print(f"  token file: {output_path}")
        print(f"  expires_at: {token.expires_at.isoformat()}")


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
    login_parser.add_argument("--project-id", required=True, help="Firebase project ID the custom token is minted for.")
    login_parser.add_argument("--uid", required=True, help="Firebase custom token UID.")
    login_parser.add_argument("--firebase-settings-key", default=None, help="Settings key passed to kiarina.lib.firebase.settings_manager.get_settings. Supplies api_key and the token_file_path the token set is written to.")
    login_parser.add_argument("--google-auth-settings-key", default=None, help="Settings key passed to kiarina.lib.google.get_credentials.")
    login_parser.add_argument("--service-account-id", default=None, help="Service account email whose iam.serviceAccounts.signBlob permission signs the custom token. Needed only for user account credentials; service account keys, impersonation, and GCE metadata each resolve a signer on their own.")
    # fmt: on

    return parser.parse_args(list(args))


extension_command_registry.register("firebase", FirebaseCommand)
