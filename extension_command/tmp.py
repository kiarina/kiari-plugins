# Usage:
#   kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/tmp.py" tmp
import argparse
from collections.abc import Sequence

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)


class TmpCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args)
        print(f"options: {options}")

        from kiarina.utils.app import user_directory

        print(f"{user_directory.get_user_data_dir()}")


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kiari ext tmp",
        description="Temporary command for testing.",
    )
    # fmt: off
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("tmp", TmpCommand)
