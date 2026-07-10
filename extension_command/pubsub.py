# RunSpec:
#   plugins:
#     - kiari_plugins/**/*.py
#
# Usage:
#   create-topic:
#     kiari ext -v pubsub create-topic --project-id my-project --topic-id my-topic
#   delete-topic:
#     kiari ext -v pubsub delete-topic --project-id my-project --topic-id my-topic
#   create-subscription:
#     kiari ext -v pubsub create-subscription --project-id my-project --topic-id my-topic --subscription-id my-sub
#   delete-subscription:
#     kiari ext -v pubsub delete-subscription --project-id my-project --subscription-id my-sub
#   publish-message:
#     kiari ext -v pubsub publish-message --project-id my-project --topic-id my-topic --attribute key=value "hello"
#   pull-message:
#     kiari ext -v pubsub pull-message --project-id my-project --subscription-id my-sub
import argparse
import asyncio
import json
import logging
from collections.abc import Sequence

from google.api_core.exceptions import AlreadyExists, NotFound  # type: ignore
from google.cloud.pubsub import PublisherClient, SubscriberClient  # type: ignore
from kiarina.lib.google import get_credentials

from kiari.cli.ext.extension_command import (
    BaseExtensionCommand,
    ExtensionCommandContext,
    extension_command_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def _parse_attributes(values: Sequence[str] | None) -> dict[str, str]:
    attributes: dict[str, str] = {}

    if not values:
        return attributes

    for entry in values:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --attribute entry: {entry!r} (expected key=value)"
            )

        key, _, value = entry.partition("=")
        attributes[key] = value

    return attributes


# --------------------------------------------------
# Main
# --------------------------------------------------


class PubsubCommand(BaseExtensionCommand):
    async def run(
        self,
        context: ExtensionCommandContext,
        args: Sequence[str],
    ) -> None:
        options = _parse_args(args, prog=self.name)

        if options.command == "create-topic":
            await self._create_topic(options)
        elif options.command == "delete-topic":
            await self._delete_topic(options)
        elif options.command == "create-subscription":
            await self._create_subscription(options)
        elif options.command == "delete-subscription":
            await self._delete_subscription(options)
        elif options.command == "publish-message":
            await self._publish_message(options)
        elif options.command == "pull-message":
            await self._pull_message(options)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {options.command}")

    # ----- topic -----

    async def _create_topic(self, options: argparse.Namespace) -> None:
        publisher = self._publisher(options)
        topic_path = publisher.topic_path(options.project_id, options.topic_id)

        try:
            topic = await asyncio.to_thread(
                publisher.create_topic, request={"name": topic_path}
            )
            print(f"Created topic: {topic.name}")
        except AlreadyExists:
            print(f"Topic already exists: {topic_path}")

    async def _delete_topic(self, options: argparse.Namespace) -> None:
        publisher = self._publisher(options)
        topic_path = publisher.topic_path(options.project_id, options.topic_id)

        try:
            await asyncio.to_thread(
                publisher.delete_topic, request={"topic": topic_path}
            )
            print(f"Deleted topic: {topic_path}")
        except NotFound:
            print(f"Topic not found: {topic_path}")

    # ----- subscription -----

    async def _create_subscription(self, options: argparse.Namespace) -> None:
        publisher = self._publisher(options)
        subscriber = self._subscriber(options)
        topic_path = publisher.topic_path(options.project_id, options.topic_id)
        subscription_path = subscriber.subscription_path(
            options.project_id, options.subscription_id
        )

        try:
            subscription = await asyncio.to_thread(
                subscriber.create_subscription,
                request={
                    "name": subscription_path,
                    "topic": topic_path,
                    "ack_deadline_seconds": options.ack_deadline_seconds,
                },
            )
            print(f"Created subscription: {subscription.name}")
            print(f"  topic: {subscription.topic}")
            print(f"  ack_deadline_seconds: {subscription.ack_deadline_seconds}")
        except AlreadyExists:
            print(f"Subscription already exists: {subscription_path}")

    async def _delete_subscription(self, options: argparse.Namespace) -> None:
        subscriber = self._subscriber(options)
        subscription_path = subscriber.subscription_path(
            options.project_id, options.subscription_id
        )

        try:
            await asyncio.to_thread(
                subscriber.delete_subscription,
                request={"subscription": subscription_path},
            )
            print(f"Deleted subscription: {subscription_path}")
        except NotFound:
            print(f"Subscription not found: {subscription_path}")

    # ----- message -----

    async def _publish_message(self, options: argparse.Namespace) -> None:
        publisher = self._publisher(options)
        topic_path = publisher.topic_path(options.project_id, options.topic_id)
        attributes = _parse_attributes(options.attribute)

        future = await asyncio.to_thread(
            publisher.publish,
            topic_path,
            options.message.encode("utf-8"),
            **attributes,
        )
        message_id = await asyncio.to_thread(future.result, options.timeout)

        print(f"Published message: {message_id}")
        print(f"  topic: {topic_path}")
        print(f"  data: {options.message}")

        if attributes:
            print(f"  attributes: {json.dumps(attributes, ensure_ascii=False)}")

    async def _pull_message(self, options: argparse.Namespace) -> None:
        subscriber = self._subscriber(options)
        subscription_path = subscriber.subscription_path(
            options.project_id, options.subscription_id
        )

        print(f"Pulling from subscription: {subscription_path}")
        print(f"  max_messages: {options.max_messages}")
        print(f"  timeout: {options.timeout}")
        print("Listening. Press Ctrl+C to stop.")

        received_count = 0

        try:
            while True:
                if options.limit is not None and received_count >= options.limit:
                    break

                try:
                    response = await asyncio.to_thread(
                        subscriber.pull,
                        request={
                            "subscription": subscription_path,
                            "max_messages": options.max_messages,
                        },
                        timeout=options.timeout,
                    )
                except Exception as e:
                    logger.error(f"Error pulling messages: {e}", exc_info=True)
                    await asyncio.sleep(5)
                    continue

                if not response.received_messages:
                    continue

                ack_ids = [msg.ack_id for msg in response.received_messages]
                await asyncio.to_thread(
                    subscriber.acknowledge,
                    request={
                        "subscription": subscription_path,
                        "ack_ids": ack_ids,
                    },
                )

                for received_message in response.received_messages:
                    received_count += 1
                    message = received_message.message
                    print("-" * 20)
                    print(f"message_id: {message.message_id}")
                    print(f"publish_time: {message.publish_time}")

                    if message.attributes:
                        print(
                            "attributes: "
                            f"{json.dumps(dict(message.attributes), ensure_ascii=False)}"
                        )

                    print(f"data: {message.data.decode('utf-8', errors='replace')}")

                    if options.limit is not None and received_count >= options.limit:
                        break

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopped.")

        print(f"Total received: {received_count}")

    # ----- clients -----

    def _publisher(self, options: argparse.Namespace) -> PublisherClient:
        credentials = get_credentials(options.google_auth_settings_key)
        return PublisherClient(credentials=credentials)

    def _subscriber(self, options: argparse.Namespace) -> SubscriberClient:
        credentials = get_credentials(options.google_auth_settings_key)
        return SubscriberClient(credentials=credentials)


def _parse_args(args: Sequence[str], *, prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"kiari ext {prog}",
        description="Manage Google Cloud Pub/Sub topics, subscriptions, and messages.",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    # fmt: off
    common_parser.add_argument("--project-id", required=True, help="Google Cloud project ID.")
    common_parser.add_argument("--google-auth-settings-key", default=None, help="Google auth settings key passed to kiarina.lib.google.get_credentials.")
    # fmt: on

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- create-topic -----
    create_topic_parser = subparsers.add_parser(
        "create-topic", parents=[common_parser], help="Create a topic."
    )
    create_topic_parser.add_argument("--topic-id", required=True)

    # ----- delete-topic -----
    delete_topic_parser = subparsers.add_parser(
        "delete-topic", parents=[common_parser], help="Delete a topic."
    )
    delete_topic_parser.add_argument("--topic-id", required=True)

    # ----- create-subscription -----
    create_sub_parser = subparsers.add_parser(
        "create-subscription",
        parents=[common_parser],
        help="Create a pull subscription.",
    )
    # fmt: off
    create_sub_parser.add_argument("--topic-id", required=True)
    create_sub_parser.add_argument("--subscription-id", required=True)
    create_sub_parser.add_argument("--ack-deadline-seconds", type=int, default=10, help="Ack deadline in seconds.")
    # fmt: on

    # ----- delete-subscription -----
    delete_sub_parser = subparsers.add_parser(
        "delete-subscription",
        parents=[common_parser],
        help="Delete a subscription.",
    )
    delete_sub_parser.add_argument("--subscription-id", required=True)

    # ----- publish-message -----
    publish_parser = subparsers.add_parser(
        "publish-message", parents=[common_parser], help="Publish a message."
    )
    # fmt: off
    publish_parser.add_argument("--topic-id", required=True)
    publish_parser.add_argument("--attribute", action="append", default=[], metavar="KEY=VALUE", help="Message attribute (repeatable).")
    publish_parser.add_argument("--timeout", type=float, default=60.0, help="Publish future timeout in seconds.")
    publish_parser.add_argument("message", help="Message body (UTF-8 text).")
    # fmt: on

    # ----- pull-message -----
    pull_parser = subparsers.add_parser(
        "pull-message",
        parents=[common_parser],
        help="Pull and immediately ack messages in a loop (for testing).",
    )
    # fmt: off
    pull_parser.add_argument("--subscription-id", required=True)
    pull_parser.add_argument("--max-messages", type=int, default=1, help="Max messages per pull request.")
    pull_parser.add_argument("--timeout", type=float, default=60.0, help="Pull request timeout in seconds.")
    pull_parser.add_argument("--limit", type=int, default=None, help="Stop after receiving this many messages. Default: no limit.")
    # fmt: on

    return parser.parse_args(list(args))


extension_command_registry.register("pubsub", PubsubCommand)
