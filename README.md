# kiari-plugins

Plugin implementations for [kiari](https://github.com/kiarina/kiari).

kiari resolves `--plugin` patterns against the local filesystem **or straight from GitHub**,
so nothing here has to be installed as a package. Point `--plugin` at a file in this
repository and its commands become available in your kiari runtime.

## Usage

```sh
kiari ext -v --plugin "@kiarina/kiari-plugins/extension_command/asr.py" asr --asr-model local ./sample.mp3
```

Load every extension command at once, and list what became available:

```sh
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
| `rtdb` | Read, write, and watch Firebase Realtime Database data |
| `pubsub` | Manage Google Cloud Pub/Sub topics, subscriptions, and messages |
| `slack` | Post / fetch / watch Slack messages (single-workspace) |
| `tmp` | Scratch command for testing the plugin path |

`rtdb` takes no token option: it resolves credentials through the
`kiarina.lib.firebase` / `kiarina.lib.firebase_rtdb` settings. Configure `api_key` and
`token_data_file_path` there, then seed the token file with `rtdb generate-token-data`.

## Optional dependencies

Most commands run on what kiari already installs. These two do not:

| Extra | Needed by |
|---|---|
| `firebase-admin` | `rtdb generate-token-data` |
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

## License

MIT
