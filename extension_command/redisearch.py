# Usage:
#   kiari ext -v redisearch migrate --schema-file ./redisearch-schema.yaml
#
#   The examples below omit the `kiari ext -v redisearch migrate` prefix:
#     --redis-settings-key hoge --redisearch-settings-key fuga --schema-file ./redisearch-schema.json
#     --schema-provider myapp.memory_schema:get_redisearch_schema
import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from kiarina.lib.redis.asyncio import get_redis
from kiarina.lib.redisearch.asyncio import (
    RedisearchClient,
    settings_manager as redisearch_settings_manager,
)
from kiarina.lib.redisearch_schema import RedisearchSchema
from kiarina.utils.common import import_object

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)


class RedisearchCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)

        if options.command != "migrate":  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

        schema = _load_schema(
            schema_file=options.schema_file,
            schema_provider=options.schema_provider,
        )
        redisearch_settings = redisearch_settings_manager.get_settings(
            options.redisearch_settings_key
        )
        redis = get_redis(
            options.redis_settings_key,
            decode_responses=False,
        )
        client = RedisearchClient(
            redisearch_settings,
            schema=schema,
            redis=redis,
        )

        existed = await client.exists_index()
        schema_changed = True

        if existed:
            current_schema = (await client.get_info()).index_schema
            schema_changed = current_schema != schema

            if schema_changed and redisearch_settings.protect_index_deletion:
                raise ValueError(
                    f"Index {redisearch_settings.index_name!r} requires migration, but "
                    "protect_index_deletion is enabled."
                )

        await client.migrate_index()

        if not existed:
            result = "created"
        elif schema_changed:
            result = "migrated (documents retained)"
        else:
            result = "unchanged"

        print("RediSearch index migration completed.")
        print(f"  index_name: {redisearch_settings.index_name}")
        print(f"  key_prefix: {redisearch_settings.key_prefix}")
        print(f"  fields: {len(schema.fields)}")
        print(f"  result: {result}")


def _load_schema(
    *,
    schema_file: str | None,
    schema_provider: str | None,
) -> RedisearchSchema:
    if schema_file is not None:
        data = _load_schema_file(schema_file)
    elif schema_provider is not None:
        provider = import_object(schema_provider)

        if not callable(provider):
            raise TypeError(f"Schema provider {schema_provider!r} must be a no-argument callable.")

        data = provider()
    else:  # pragma: no cover - argparse enforces the mutually exclusive group
        raise ValueError("Either schema_file or schema_provider is required.")

    return _normalize_schema(data)


def _load_schema_file(file_path: str) -> Any:
    path = Path(file_path).expanduser()
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported schema file extension {suffix!r}. Use .json, .yaml, or .yml.")


def _normalize_schema(data: Any) -> RedisearchSchema:
    if isinstance(data, RedisearchSchema):
        return data

    if isinstance(data, list):
        return RedisearchSchema.model_validate({"fields": data})

    if isinstance(data, dict):
        return RedisearchSchema.model_validate(data)

    raise TypeError(
        "Schema must be a RedisearchSchema, a field list, or an object with a fields key."
    )


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    examples = """Examples:
  kiari ext redisearch migrate --schema-file ./redisearch-schema.yaml
  kiari ext redisearch migrate \\
    --redis-settings-key hoge \\
    --redisearch-settings-key fuga \\
    --schema-file ./redisearch-schema.json
  kiari ext redisearch migrate \\
    --schema-provider myapp.memory_schema:get_redisearch_schema"""

    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Manage RediSearch indexes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Create or migrate an index while retaining its Redis Hash documents.",
        description="Create or migrate a RediSearch index to the supplied schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    # fmt: off
    migrate_parser.add_argument("--redis-settings-key", default=None, help="kiarina-lib-redis settings key. Uses the default key when omitted.")
    migrate_parser.add_argument("--redisearch-settings-key", default=None, help="kiarina-lib-redisearch settings key. Uses the default key when omitted.")
    schema_group = migrate_parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument("--schema-file", help="JSON or YAML schema file containing a field list or a fields object.")
    schema_group.add_argument("--schema-provider", help="No-argument schema provider import path in module:object format.")
    # fmt: on
    return parser.parse_args(list(args))


extension_command_registry.register("redisearch", RedisearchCommand)
