# TranscriptViewer Web（Cloudflare版）

ネイティブ版 TranscriptViewer とほぼ同じ機能をWebで提供する非公開Worker。個人の配信録画を
扱うため、**X（OAuth 2.0 + PKCE）でログインした許可済みアカウントだけ**が閲覧できます。
秘密の値と録画由来のデータは Git に入れません（下の「Git管理しないもの」）。

## 構成

| 場所 | 役割 |
| --- | --- |
| `public/index.html` | 画面（配信者一覧・動画一覧・カレンダー・検索・設定） |
| `src/worker.js` | Worker（X認証、ルーティング、R2配信、許可リスト、デバッグ用セッション） |
| `src/upload-worker.js` | 大容量メディア用の一時アップローダー（同期中だけデプロイして削除） |
| `src/sync/*.ts` | 同期処理（TypeScript）。変更検出・計画・アップロード |
| `src/bin/*.ts` | コマンドの入口（`sync` / `deploy` / `debug`） |
| `tools/build_catalog.py` | 配信ライブラリからカタログと字幕データを生成 |
| `cloudflare.config.ts` / `wrangler.config.ts` | Worker とバインディングの設定 |

R2バケットには `catalog.json`、`data/<id>.json`（字幕・保存済み要約）、`media/`（編集済み動画）、
`media-original/`（元動画。MKVはH.264プロキシ）、`config/allowed-users.json`（許可リスト）を置きます。

## URL

- `/` → `/streamers`（配信者一覧）
- `/streamers/<配信者>`（動画一覧。`?date=YYYY-MM-DD` で日付絞り込み）
- `/streamers/<配信者>/videos/<動画ID>`（動画詳細）
- `/admin`（管理者のみ。閲覧を許可するXアカウントの追加・削除）

## コマンド

```sh
bun install --frozen-lockfile
bun run tsc        # 型検査
bun run lint       # biome
bun run test       # vitest（ファイルごとに90%以上のカバレッジ）
bun run sync -- --dry-run   # 変更されたものを一覧するだけ
bun run sync                # 変更分だけR2へ反映
bun run deploy              # Worker本体のデプロイ（画面・Worker変更時）
bun run debug /api/me       # 管理者の読み取り専用セッションで認証後ページを取得
```

## 同期（`bun run sync`）

動画を再編集したり、要約・字幕を作り直したりしたら `bun run sync` を実行します。

1. 配信ライブラリからカタログを再生成します（`data.next/`）。
2. **動画ファイルだけでなく周辺ファイルもすべて指紋化**します: 編集済み動画・元動画
   （サイズ＋更新時刻）、字幕、保存済み要約 `<字幕>.summaries/current.json`、字幕と同じ
   ディレクトリにある同名ファイル（ハッシュ）。生成された `data/<id>.json` のハッシュも見ます。
3. 前回アップロード時の記録 `sync-state.json` と比べ、変わったものだけを理由つきで一覧します
   （例: `source files changed: summary`）。記録がなく、バケットに同サイズの動画がある場合は
   アップロード済みとして取り込みます（MKV元動画はローカルのプロキシのサイズと比較）。
4. 大きいメディアは一時アップローダーWorkerで送ります。実行ごとに新しいランダムトークンで
   デプロイし、**成功・失敗にかかわらず終了時に削除**します。続いて `data/<id>.json`、最後に
   `catalog.json` を `cf r2 objects put` で送ります（カタログが未送信の動画を指さないため）。
5. `--data-only` はメディアとカタログを送りません。

設定は `sync.config.json`（Git管理外。`sync.config.example.json` が雛形）で、配信者名・
ライブラリのルート・バケット名を指定します。同期ではWorker本体を再デプロイしません。

## 要約

要約は `transcript-summary generate`（端末内のApple Intelligence。リポジトリのREADME参照）で
字幕の隣に保存されます。`uv run python -m replay.viewer_summary_publish` は保存済み要約だけを
ビューアのデータへ写してアップロードします。同期はその要約の更新も検出します。

## 認証と許可リスト

- ログインは X の OAuth 2.0（PKCE）。管理者は環境変数 `X_ADMIN_USERNAMES`、そのほかの閲覧許可は
  管理者が `/admin` で追加し、R2の `config/allowed-users.json` を正本、KVとCache APIを読み取り層にします。
- 必要なWorkerの値は `secrets.json` に置きます（雛形は `secrets.example.json`）:
  `X_CLIENT_ID`, `X_CLIENT_SECRET`, `SESSION_SECRET`, `X_ADMIN_USERNAMES`。カスタムドメインは
  `deploy.env` の `VIEWER_DOMAIN`（雛形は `deploy.env.example`）。どの値をどこに置くかは
  リポジトリの `docs/secrets-and-environment.md` を参照。

## デバッグ用の管理者セッション（読み取り専用）

- `POST /auth/debug` にヘッダー `X-Debug-Token` を付けると、1時間有効・GETのみの管理者セッション
  Cookieを発行します。トークンが違う・未設定・期限切れ（`DEBUG_TOKEN_EXPIRES`）なら常に404です。
- 値は `secrets.json` の `DEBUG_TOKEN`（32文字以上）と `DEBUG_TOKEN_EXPIRES`（UNIX秒）。
- 無効化は `secrets.json` から2項目を消して `bun run deploy`。

## Git管理しないもの（`.gitignore`）

`secrets.json`、`credentials.txt`、`deploy.env`、`sync.config.json`、`.debug-cookies`
（秘密・個人設定）、`data/` と `data.next/`（録画の字幕・要約）、`proxies/`、`sync-state.json`、
ログ、`node_modules/` など。リポジトリのルートは MPEG-TS 動画のため `*.ts` を無視しますが、
このディレクトリでは TypeScript を追跡できるよう `!*.ts` で許可しています。
