# kiari-plugins

Plugin implementations for [kiari](https://github.com/kiarina/kiari).

kiari resolves `--plugin` patterns against the local filesystem **or straight from GitHub**,
so nothing here has to be installed as a package. Point `--plugin` at a file in this
repository and its commands become available in your kiari runtime.

## Setup

Do this once, and every command in this repository is available as a plain
`kiari ext <command>`. The `# Usage:` block in each plugin file assumes you have.

Create a profile and add this repository's `extension_command/` directory to the RunSpec
it creates (`~/.config/kiari/profiles/plugins/run_spec.yaml`):

```sh
kiari profile new plugins
```

```yaml
plugins:
  - "@kiarina/kiari-plugins/extension_command/"
```

Make it the current profile:

```sh
kiari profile use plugins
```

From here on, no flags are needed:

```sh
kiari ext asr --asr-model local ./sample.mp3
```

To leave your current profile alone, skip `profile use` and select it per run with `-p`:

```sh
kiari ext -p plugins asr --asr-model local ./sample.mp3
```

## Usage without a profile

A single file, or the whole directory, loaded straight from GitHub:

```sh
kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/asr.py" asr --asr-model local ./sample.mp3
kiari ext --plugin "@kiarina/kiari-plugins/extension_command/"
```

Notes:

- Options must come before the command name. `kiari ext` passes everything after the
  command name through to the command untouched.
- The first time you reference `@kiarina/...`, kiari asks you to trust the GitHub user.
  Resolved files are cached locally afterwards.
- Append `@<commit-hash>` to pin a revision, e.g.
  `@kiarina/kiari-plugins/extension_command/asr.py@a1b2c3d`. Without it, `main` is used.
- Every plugin file starts with a `# Usage:` comment block holding its own examples.

## Updating the plugins

**A cached directory is served as-is: kiari does not re-check GitHub for it.** Once
`@kiarina/kiari-plugins/extension_command/` has been resolved, later runs list the files
already in `~/.cache/kiari/github_files/` and never learn about files added upstream.
A command you know exists here simply will not be found.

To pick up changes, clear the cache and let the next run re-fetch:

```sh
kiari admin clear-cache -f
```

That deletes kiari's whole cache directory, not just this repository's files. It is a
cache, so nothing is lost beyond the next run being slower. Drop `-f` to see what would
go and confirm first.

`--github-ignore-cache` (or `KIARI_GITHUB_IGNORE_CACHE=1`) re-fetches for one run without
clearing anything:

```sh
kiari ext --github-ignore-cache asr --asr-model local ./sample.mp3
```

Prefer `clear-cache` when a plugin was **deleted or renamed** upstream. Re-fetching adds
and overwrites but never prunes, so a stale file keeps registering its command and can
shadow a rename long after it is gone from this repository.

Pinning with `@<commit-hash>` does not sidestep any of this — the cache path ignores the
revision, so a pinned spec and `main` share one directory.

## Extension commands

`extension_command/` adds subcommands to `kiari ext`.

| Command | What it does |
|---|---|
| `asr` | Transcribe audio with kiarina_agi ASRModel |
| `tts` | Generate speech with kiarina_agi TTSModel |
| `vad` | Detect voice segments with kiarina_agi VADModel |
| `scd` | Detect speaker changes with kiarina_agi SCDModel |
| `audio-tagging` | Tag an audio file with kiarina_agi AudioTaggingModel |
| `audio-embedding` | Create, list, and search audio embeddings |
| `image-detection` | Detect faces / objects with kiarina_agi ImageDetectionModel |
| `image-embedding` | Create, list, search, and validate image embeddings |
| `video-source` | Inspect frames emitted by kiarina_agi VideoSource |
| `text-embedding` | Create, list, search, and validate text embeddings |
| `firebase` | Authenticate against Firebase (`login`) |
| `firebase-rtdb` | Read, write, and watch Firebase Realtime Database data |
| `google` | Authenticate a Google user account and output its credentials |
| `pubsub` | Manage Google Cloud Pub/Sub topics, subscriptions, and messages |
| `slack` | Post / fetch / watch Slack messages (single-workspace) |
| `tmp` | Scratch command for testing the plugin path |

`firebase-rtdb` takes no token option: it resolves credentials through the
`kiarina.lib.firebase` / `kiarina.lib.firebase_rtdb` settings. Configure `api_key` and
`token_file_path` there, then seed the token set with
`firebase login --project-id ... --uid ...`, which writes it to that same
`token_file_path`.

`google` resolves a `user_account` configuration from `kiarina.lib.google`. Pass
`--google-settings-key` to select a named configuration, or omit it to use the default.
It always prints the authorized user credentials JSON and also saves it to
`authorized_user_file` when that setting is configured.

## Optional dependencies

Most commands run on what kiari already installs. These commands do not:

| Extra | Needed by |
|---|---|
| `firebase-admin` | `firebase login` |
| `google-auth-oauthlib` | `google` |
| `imageio` | `video-source --output-video` |

Install the extra into the same environment as kiari — for local development,
`uv sync --all-extras`.

## Development

```sh
make          # format, then lint
make format   # ruff check --fix + ruff format
make lint     # ruff check + ruff format --check + mypy
make upgrade  # bump the lockfile, then run make lint to catch API drift
```

ruff and mypy are configured to match kiari; mypy runs in `strict` mode.

The dev environment takes **kiari and kiarina from the HEAD of their default branch**, not
from PyPI (see `[tool.uv.sources]`), so breaking changes surface here before they reach a
release. `uv.lock` pins the commit, so this stays reproducible; `make upgrade` moves it
forward. The `>=` specifiers under `[project]` remain the declared contract — pip and
anything consuming this repository ignore `[tool.uv.sources]` entirely.

## License

MIT
