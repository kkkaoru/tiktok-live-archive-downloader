# TikTok LIVE replay downloader

Mac上で、アクセス権のある終了済みLIVE録画を直接ダウンロードするCLIです。画面録画・等倍再生での保存は行いません。非公式APIのため、TikTok側の変更やアカウント権限によって利用できない場合があります。

## セットアップ

新しいMacでは、Homebrewを公式の https://brew.sh から導入してPATHを設定したあと、以下を実行します。Apple Silicon / IntelでHomebrewのインストール先を固定しない構成です。

```sh
git clone https://github.com/kkkaoru/tiktok-live-archive-downloader.git
cd tiktok-live-archive-downloader
bash scripts/bootstrap.sh
```

Python **3.12.13** を `.python-version`、Python依存関係を `uv.lock` で固定し、`uv sync --locked` で構築します。スクリプトはuv・FFmpeg・ChromeをHomebrewから導入します。既存の手動インストール済みChromeを使う場合や、数値IDだけを使う場合は `--no-browser` でChromeのインストールを省略できます。

Homebrew自体のインストーラー、ログイン、動画取得は自動実行しません。HomebrewのFFmpeg・Chrome等のバージョンは実行時点のものなので、システム全体のビット単位の再現を保証する構成ではありません。認証情報はリポジトリに含まれず、別途設定が必要です。

認証は環境変数 `TIKTOK_SESSIONID`（32桁のセッショントークン）。値をチャット、シェル履歴、Gitへ直接記載しないでください。秘密ファイルやシークレット管理ツールから設定します。

```sh
read -r TIKTOK_SESSIONID < private/tiktok-sessionid.secret
export TIKTOK_SESSIONID
```

数値ID指定の一覧・動画取得とCookie再発行はMacとhttpxだけで動き、Android、ADB、Frida、ブラウザーは不要です。ユーザー名の解決では、プロフィールがHTTPに情報を返さない場合だけ、オプションの非表示Chromeを使用します（`uv sync --extra web`、Chromeのインストールが必要）。

## 対象ユーザーを実行時に指定

表示名ではなく、ユーザー名（`@handle`）、正式なプロフィールURL、または確認済みの数値ユーザーIDを指定します。

```sh
uv run replay recordings --user @creator
uv run replay recordings --user 'https://www.tiktok.com/@creator'
uv run replay recordings --anchor-id 123456789

uv run replay download-all downloads --user @creator --workers 6 --max-gib 12
```

`--user` と `--anchor-id` は同時指定不可。ユーザー名は認証情報を送らない独立したHTTPS接続でプロフィールを取得し、返されたユーザー名と数値IDを照合します。短縮URL、動画URL、他ホストへのリダイレクト、曖昧なプロフィール応答は拒否します。HTTPにプロフィール情報がない場合は、新規コンテキストの非表示Chromeに自動で切り替えます。既存ブラウザー、アカウント認証、Macの入力・フォーカスは使わず、動画リクエストを遮断します。CAPTCHA等は解かず、通常ページで確認できない場合は停止するため、確認済みの `--anchor-id` を指定してください。

定期実行などでは環境変数で対象を指定できます。CLI引数を指定した場合は、そちらが優先されます。

```sh
export TIKTOK_TARGET_USER='@creator'
uv run replay recordings
uv run replay download-all downloads --workers 6 --max-gib 12 --refresh-session
```

対象指定はアクセス権を与える操作ではありません。**ログインしたアカウントが閲覧できる録画通知**を最後まで列挙し、対象の数値ユーザーIDと一致する実在の配信だけを取得します。通知にない配信・他アカウント限定録画は取得できません。

## 重複防止と互換性

- 安定した配信IDで重複判定。完了済み・他プロセスが取得中の配信は再取得しません。
- `private/jobs/` が完了記録です。削除しないでください。既存の出力や不整合を検出した場合、勝手に上書きせず検証を要求します。
- 新規保存先は `downloads/replay-配信ID.mp4`。署名付きURL・認証値は一覧に出力しません。
- HLSを並列取得し、再エンコードせずMP4に格納。HEVCはQuickTime向け `hvc1` タグとfaststartを使用します。
- `--workers` は1〜16、`--max-gib` は1配信あたりの上限です。
- `--store` はグローバル引数です。例：`replay --store private/candidates recordings --user @creator`。親ディレクトリを変えると完了記録と更新トークンの保存先も変わります。

## Androidなしのセッション処理

```sh
uv run replay refresh-session
uv run replay recordings --user @creator --refresh-session
```

正規Web SDKと同じCookie再発行要求をHTTPで実行し、同一アカウント・通知API利用を検証してから秘密ファイルへ原子的に保存します。通常は人の操作もブラウザー表示も不要です。単独コマンドの保存先は `--output` で指定できます。

**Cookie再発行は、有効期限の延長や失効後の復旧と同義ではありません。** 実測済みなのは有効なセッションの再発行とAPI利用です。結果の `reissued` / `rotated` と `expiry_extended` で違いを表示します。比較情報がない場合は `null`。Cookie期限はサーバー側の認証有効期間を保証しません。

新しいWebログインが必要な場合のみ、明示的に実行します。

```sh
uv sync --extra web
uv run replay refresh-session --login
```

新しい専用Mac Chromeで通常ログインを行います。既存プロファイル・Macの入力自動操作は使用しません。認証コード等は本人確認が必要です。この経路は自動テスト済みですが、実アカウントの新規Webログインと新規Webトークンでの録画取得は未実証です。

更新後の別プロセスでは秘密ファイルから環境変数を再読み込みしてください。`--refresh-session` 指定中の取得処理は、更新結果を直接利用します。子プロセスは親シェルの環境変数を変更しません。

## 保存先・安全性・開発

[ディレクトリ構成](docs/architecture.md)と[公開・移行時の安全性](SECURITY.md)を参照してください。ローカルの動画、認証、実アカウントの調査記録はGit管理しません。別Macへの動画移行では、既存の完了記録に絶対パスが含まれる点にも注意してください。

```sh
bash scripts/bootstrap.sh --dev --no-browser
make check
# 公開予定の変更だけを確認・ステージしたあと：
make public-check
```

Ruff format/lint、strict mypy、pytest、ファイル別カバレッジ90%以上を検証します。追加のGitleaks検査はステージ内容と全参照履歴を対象とし、ローカルの秘密情報や動画をスキャン先へコピーしません。通常のテストは実アカウント・ネットワーク・Androidを必要としません。

macOS CIも同じロックファイルで検証し、アカウントのシークレットを要求しません。GitHub ActionsはコミットSHA固定・読み取り権限のみです。wheel/sdistのビルド対象はソースと公開ドキュメント等に明示限定しています。
