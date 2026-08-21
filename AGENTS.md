# AGENTS.md

このリポジトリで作業するエージェント向けのガイドラインです。

## このリポジトリは何か

kiari の plugin 実装の公開置き場です。パッケージとして配布することは目的ではなく、
**kiari が `--plugin` で GitHub から直接ロードできる状態を保つこと**が目的です。

kiari の registry へ register できるものなら種類を問わず置けます（agent、chat_provider、
schedule_handler、watch_handler など）。現在あるのは `extension_command/` だけです。
新しい family を足すときは、registry 名と同じディレクトリ名にしてください。

## 作業前に読むもの

- `README.md`
- `pyproject.toml`
- `Makefile` / `mise.toml` / `.mise/tasks/`
- 触る plugin ファイル冒頭の `# Usage:` コメント

kiari 本体の plugin 機構・registry・RunSpec の正典は kiari リポジトリの
`docs/concepts/runtime-configuration-and-extensibility.md` です。

## plugin を追加するとき

1. family ごとのディレクトリへ `.py` を 1 ファイル置く
2. 冒頭に `# Usage:` コメントを書く。1 行目は `kiari ext -v <name> ...` の形式で、
   2 行目以降は prefix を省略して引数だけを並べる。`--plugin` は書かない
   （README の Setup にある profile 経由のロードを前提とする）
3. import 時の副作用として registry へ register する
4. kiari が既に依存していない依存を使うなら `[project.optional-dependencies]` へ extra を足し、
   README の表にも 1 行足す
5. `make` で format と lint を通す

## コードの規約

- ruff / mypy の設定は kiari 本体と揃えています。mypy は `strict = true`
- 引数は argparse で自前にパースします。kiari は `kiari ext <name>` 以降を素通しします
- 型スタブのないライブラリは `# type: ignore[<code>]` と code 付きで書きます
- jaxtyping の shape 文字列は forward reference と解釈されるため、
  `X: TypeAlias = UInt8[np.ndarray, "..."]  # noqa: UP040, F722` の形にします
  （kiarina-python と同じ書き方）

## 依存を更新するとき

**kiari / kiarina は PyPI ではなく、それぞれの default branch の HEAD から取得します**
（`pyproject.toml` の `[tool.uv.sources]`）。リリース前に drift を拾うためです。
`uv.lock` には commit が固定されるので、HEAD へ追随するのは `make upgrade` を叩いたとき
だけです。`project.dependencies` の `>=` 指定は宣言として残してあり、外部から見える契約は
そちらです。

`make upgrade` で lockfile を更新し、`make lint` を通してください。kiari / kiarina には
破壊的変更が入ることがあり、mypy がその多くを検出します。壊れた plugin を放置せず、
同じコミットで新しい API へ追従させます。

ただし mypy を通り抜ける破壊的変更もあります。argparse の `Namespace` は `Any` 扱いなので、
設定クラスからフィールドが消えても検出されません。設定を読むコマンドを触ったら、
`--help` と必須引数エラーまで実際に流してください。

kiarina は monorepo（uv workspace）ですが、source 指定は `packages/kiarina` の 1 つで
足ります。uv が checkout から workspace root を見つけ、member 全部を同じ commit から
解決します。`kiarina-falkordb` だけは member ではないため PyPI のままです。

まだ push していない変更を試すときだけ、一時的に `path` source へ差し替えます
（他マシンでは壊れるので、コミットには含めません）。

## コミットするとき

**次の作業は別の担当者に引き継がれる**前提で作業してください。未検証の懸念や次の一手が
あれば、コミットメッセージか該当ファイルのコメントに残してください。
