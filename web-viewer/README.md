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
- `/s/<トークン>`（共有リンク。**ログイン不要な唯一の入口**。下の「共有リンク」）

## 共有リンクとダウンロード

- 管理者は動画ページの「共有」で、編集済み・元動画のどちらか／両方を選んでリンクを発行します。
  トークンは発行ごとの32バイト乱数（URL上43文字）、表示期間は既定7日（1〜365日で指定）。
- 記録はR2の `shares/<トークン>.json`。動画の表示用項目と許可した素材だけを複製して持ち、カタログ
  （ローカルパスを含む）は共有側に出しません。期限切れ・無効・削除済みはすべて同じ404です。
- 発行済みリンクは同じダイアログか `/admin` の「オフにする」で即時に無効化（記録を削除）します。
- 共有ページは **Cloudflare Turnstile** でボットを拒否します。確認に通ると、その共有URLだけに有効な
  署名付きCookie（12時間）が発行され、ページ・データ・動画はそのCookieがある場合だけ返します。
  キーは `secrets.json` の `TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY`（ウィジェットは
  workers.dev とカスタムドメインを許可）。キーが無いと共有は閉じたまま（503）です。`/robots.txt` は全拒否。
- リンクプレビュー（OGP）: 確認画面に配信者・配信日時・長さ・期限のmetaを入れ、`/s/<トークン>/og.jpg`
  だけはTurnstileなしで返します（クローラーは確認を通れないため）。画像は
  `uv run python -m replay.share_preview --catalog web-viewer/data/catalog.json --bucket <バケット>` が
  盛り上がりのピーク位置のフレームから1200×630で作り、R2の `og/<id>.jpg` へ変更分だけ送ります。
  文字起こし・要約・動画はプレビューに含めません。
- 共有ページはその1本だけを表示し、要約・認識結果のジャンプと、その動画内の検索ができます。
- ダウンロードは「動画データ」（R2から直接・Range対応で再開可能、`?download=1`）と
  「文字起こし・要約」（テキスト1ファイル、ブラウザ内で生成）の2つ。数GBの動画をWorkerで
  まとめてZIPにするとメモリ・CPU制限を超えるため分けています。

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

## 盛り上がり

`uv run python -m replay.excitement catalog --catalog web-viewer/data/catalog.json --fetch-chat`
（リポジトリ直下で実行）が字幕の隣に `<字幕>.excitement.json` を作り、同期で `data/<id>.json` の
`excitement` として載ります。計算はこのPythonだけが行い、Web版とMac版TranscriptViewerは表示のみです。

- 発話: 区間ごとの認識文字数と、発話中の音量（dBFS）。YouTubeでは区間ごとのコメント数とコメント文字数。
- どれも**その動画内の順位（パーセンタイル）**に変換して平均し、ピークを1に揃えた相対値です。文字数・dB・
  件数は補足としてホバー／クリック時に表示します。合算と別々（発話・コメント）は画面で切り替えます。
- 再計算しない: 音量は動画の内容キー（サイズと先頭・末尾1MiB）ごと、コメントはYouTube IDごとに
  `edits/excitement-cache/` へ一度だけ保存。字幕が変わっても音量は再デコードしません。出力は入力が
  変わったときだけ書き換わり、同期も変わったときだけ送ります。Mac版は字幕ハッシュが一致しない古い
  推定を表示しません。
- コメントの時刻は元動画基準です。無音カット済みの編集版字幕ではずれるため、元動画の時刻で再生する
  動画だけに使います。コメント本文・投稿者名は公開データに含めません（件数と文字数のみ）。

## 切り抜き

動画ごとに、盛り上がりと文字起こしから切り抜き候補を作り、R2に保存します。**既定ではMacで作り**
（`bun run clips:mac`、下記）、Cloudflare上のAI（Workers AI）は**ユーザーが再計算を指示したときだけ**動きます。

- Macでの作成（既定）: `bun run clips:mac -- <videoId|data.json>...` が `web-viewer/data/<id>.json` から
  同じ `generate()` を `place: "mac"` で動かし、Bearerの `PUT /api/clips/<id>.json` で保存します。
  clefの段は `mlx-community/clef-OptiQ-4bit`、タイトル・要約はGLMの代わりに `Ternary-Bonsai-2-9B` が、
  `python -m replay.clip_models`（127.0.0.1:8791、`uv run --isolated --locked --extra clips` で共有の
  `.venv` を変えずに起動、終わったら停止）で答えます。clairicはCloudflareのモデルなのでMacでは読まず、
  出荷判定前のふるい（clef-flash）も行いません（clairicが無いとふるいは何も落とさないため）。clefだけの
  回帰で出荷を決めます。記録が無い動画と、データが変わった動画だけを作ります（MacとCloudflareのどちらで
  作った記録でも、同じデータからなら作り直しません）。Workerは保存時にmovieKeyを公開中の動画から求め直します。
  録画パイプラインの `clips` 工程はこれを1台で1本ずつ（`clips-mac.lock`）実行し、空きメモリが40%未満なら
  モデルサーバーは起動を拒みます（2モデルで約22GB）。

- 長さは合計 **10〜30秒（短尺）か30〜90秒（長尺）** のどちらか。1つの切り抜きは同じ配信の**複数の範囲**を
  含められ、再生すると範囲を順に続けて再生します（Web版・Mac版とも）。秒数と各範囲の時刻を表示します。
- 構成は次の2つのどちらか。どちらも**最初の3秒が強い引き**であることを必須とし、最初の範囲は候補の中で
  最も引きの強い発話から始めます。
  - **山場→経緯**: 一番の盛り上がりを最初に流し、その後にそこへ至る経緯を時系列で流す。
  - **引き→山場**: 疑問や気になる起点から始まり、最後の範囲が一番の盛り上がりになる。
- **1つの切り抜きは1つの話題だけ**。同じ配信の切り抜き同士は範囲が重ならず、開始・終了も共有しません。
- 作り方（`src/clips.ts`、アルゴリズム版7）:
  1. 同期（`bun run sync`）が `python -m replay.clip_terms` で各発話を形態素解析し（固定版Vibrato辞書）、
     `data/<id>.json` の字幕に内容名詞 `terms`、「文の途中から始まらないか」`opens`、「格助詞・接続助詞・係助詞
     （を・て・けど・は…）で文が途切れて終わらないか」`closes` を加えます。
  2. 盛り上がりのピーク（最大18か所、90秒以上離す）ごとに前120秒〜後30秒の発話を、間（3秒）・最大10秒・
     5秒以降の文頭で区切った単位にします。範囲は文頭からしか始めません。
  3. 名詞の重なり（前後30秒の余弦類似度）が低い文頭を話題の切れ目候補にし、`@cf/cloudflare/clef` が
     切れ目かを判定します（12秒以上の無音も切れ目）。山場を含む話題の区切りの中だけを使います。
     clefへの状態には「話題」の定義（コメント読み・あいさつ・お礼・配信の進行・別の対象の雑談は別の話題）を
     毎回含めます。
  4. 同じ問い合わせで各単位の「面白いか」「疑問・起点になるか」「区切りで話がオチ・結論・反応に達するか」
     「山場と同じ話題か」（山場の発話を引用して問い、名詞の一致度とclefの高い方）と、冒頭候補ごとの
     「最初の3秒が強い引きか」を問います。山場が話の途中で終わる場合は、話が区切れる単位まで
     （同じ話題・20秒以内）山場を延ばします。
  5. 2つの構成を短尺・長尺で組み立て、範囲の端を文の切れ目に合わせます（文の続きを含め、
     重なった字幕は最後まで流す）。clefに1本ずつ（時系列順に）「同じ一つの話題か」「無関係な発話が
     混ざっていないか」「これだけで分かるか」「話が区切れて終わるか」と、発話ごとに「主題と無関係か」を
     問います。無関係な発話が大半を占める単位（山場と冒頭以外）と範囲の端の無関係な発話を外して
     読み直し、最大2回まで直します。話題がまとまらないもの、話の途中で終わるものは採用しません
     （別系統の採点モデルで130候補を測ると、clefが「区切れて終わらない」と読んだ候補はまとまりに関係なく
     良い切り抜きになりにくかったため）。
  6. 最後に流れる発話が `closes` でない（文が途切れて終わる）もの、まとまり0.2未満のものは候補にしません。
     残りを点数（引き・面白さ・盛り上がり・まとまり・分かりやすさ・終わり方）順に16本まで、clefが
     採点モデルと同じ基準（話題・分かりやすさ・引き・終わり方・流れの5段階と、良い切り抜きか・前提があるか・
     途中から始まるか・途中で終わるか・関係ない発話が混ざるかの判定）でもう一度読みます。その答えを、
     採点モデルで採点済みの191本（30配信）に当てはめたロジスティック回帰に通し、採点モデルが「良い」と
     判定する確率を出します。`@cf/cloudflare/clairic` も同じ基準を5段階の分布（平均）として読み、オチになる
     発話がない確率と合わせて同じ回帰に加えます（clairicが失敗したらclefだけの回帰に戻します）。
     0.7以上のものだけを、確率の高い順に、範囲が重ならないよう短尺6本・長尺4本まで残します（配信ごとに外して
     検証すると、0.7以上は84%が良い切り抜き・44本、Wilson下限0.71。clefだけでは81%・37本、0.66）。
     `@cf/zai-org/glm-5.3-flash` は残った切り抜きのタイトル・要約を付けるだけで、1動画につき1回、
     残るものがない動画や前回と同じ切り抜きだけの動画では呼びません。返答が形式外なら3回まで繰り返します。
- プロンプト（`src/prompts.ts`）: モデルに送る指示はすべてi18nと同じ形式のカタログで管理します。キーごとに
  日本語 `ja` と英語 `en` を並べ、`{名前}` の差し込みは両方で同じにします（テストで検査）。送る言語は
  `LANGUAGE` で選び、今は日本語です（英語では採点済み191本でトークンが減らず、精度も上がらなかったため）。
  日本語の文言はバージョン9と1バイトも違わず、同じ動画には同じリクエストを送ります。clefは同じ
  リクエストには同じ答えを返しますが、同じ質問でも一緒に聞く質問や言い回しが変わると答えが変わります
  （質問を短くしたり、区間の質問を2回に分けたりすると、検証用の2配信で良い切り抜きが減りました）。
  そのため文言・質問の組み合わせ・区切り方を変えたら、`bun run clips:eval` で良い切り抜きの本数を
  確かめてから `ALGORITHM` を上げます。1回の生成の中で同じclefリクエストを2度送ることになった場合
  （作り直した候補が既に読んだ候補と同じになった、同じ候補が2つ並んだ）は、最初の答えを使います
  （16配信で結果は同じまま、1動画あたり約6回減）。区間は山場を含むリクエストから先に送り、その中に
  引きが1.8（0〜4）に届く冒頭が一つもなければ残りは送りません（16配信288区間で、これを下回る区間から
  候補は一つも出ていません）。出荷判定の前に `@cf/cloudflare/clef-flash` が同じ採点基準で読み、
  clairicと合わせた見込みが0.2未満の候補はclefに読ませません（採点済み191本で、clefが出荷する44本を
  一本も落とさない下限）。clef-flashは出荷を決めません。区間・読み直しをclef-flashやclef-omniに
  替える案、読み直しを両者の平均にする案は、検証用の2配信で良い切り抜きが減ったため使っていません。1動画ごとに、モデル別の実際の呼び出し回数と
  入力トークンを `{"clipsUsage": ...}` の1行でWorkerのログに出します。
- 精度の評価（`src/clips-eval.ts`、`bun run clips:eval`）: 出荷する `generate()` をそのままローカルで動かし
  （R2には書き込みません）、規則（長さ・構成・重なり）の検査、範囲の端が文の切れ目かの計測、
  パイプラインとは別系統のモデル `@cf/deepseek-ai/deepseek-v4.1-flash-1017` による5項目
  （話題の一貫性・単独での分かりやすさ・引き・終わり方・流れ）の採点をします。採点モデルは、本物の切り抜きに
  無関係な区間を差し込んだものや端を文の途中で切ったものを低く採点できるかで先に較正します。
  モデルの応答は入力ごとにキャッシュし、変わった段だけを呼び直します。例:
  `bun run clips:eval -- --label v5 --sets SETS.json --out DIR --cache DIR [--records 保存済みrecord] [--previous 要約を再利用するrecord] [--calibrate]`
  （`SETS.json` は `{"dev": [id...], "heldOut": [id...]}`。調整に使った動画と未使用の動画を分けて報告します）。
- 再生: 「🔁 再生」（既定）は範囲を順に流し、最後の範囲の後に最初へ戻って繰り返します。範囲ボタンもループです。
  「▶ 1回再生」は1回で止まります（範囲の外へ移動すると切り抜き再生をやめます）。Mac版も同じです。
- キャッシュ: R2の `clips/<movieKey>.json`。movieKeyは動画のサイズと先頭・末尾1MiBのSHA-256
  （`replay.excitement.movie_key` と同じ）で、Workerは動画をダウンロードせずに範囲読み取りで求めます。
  字幕・盛り上がり・アルゴリズム版・モデルのハッシュが同じなら再計算しません。再計算でも範囲が同じ切り抜きの要約は再利用します。
- Cloudflareでは自動実行しない: R2の `data/*.json` の保存通知はキュー `transcript-viewer-clips` に入りますが、
  Workerの `queue()` は動画からmovieKeyへの対応を更新するだけで、AIを呼びません。未計算の動画を開いても
  キューに入れず `missing` を返します。
- 再計算（ユーザーの指示、Cloudflare AIを使う唯一の経路）: Web版の動画ページの「切り抜きを再計算」
  （管理者のみ）、Mac版の同名ボタン、または `POST /api/clips/<id>/reload`。計算中は画面が15秒ごとに確認します。
- API: `GET /api/clips/<id>.json[?key=<movieKey>]` → `ready`（record付き）/`pending`/`missing`/`mismatch`
  （渡したmovieKeyが公開中の動画と違う）。ログイン済みセッションのほか、`secrets.json` の
  `CLIPS_API_TOKEN` をBearerで送るとこのAPIだけ使えます（Mac版用）。共有リンクでは読み取りのみ。
  `PUT /api/clips/<id>.json`（Bearerのみ）はMacで作ったrecordを検証して保存し、`written`/`invalid`(400)/
  `no-data`・`no-movie`(404) を返します。
- 初回の設定（済み）: `cf queues create --queue-name transcript-viewer-clips`、
  `cf r2 buckets event-notifications update <queue-id> --bucket-name transcript-viewer-media --rules '[{"actions":["PutObject","CompleteMultipartUpload","CopyObject"],"prefix":"data/","suffix":".json"}]'`、
  `bun run deploy`。

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
