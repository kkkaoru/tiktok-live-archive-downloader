# TikTok LIVE replay downloader

Mac上で、アクセス権のある終了済みLIVE録画を直接ダウンロードするCLIです。画面録画・等倍再生での保存は行いません。非公式APIのため、TikTok側の変更やアカウント権限によって利用できない場合があります。

## ドキュメント

- [運用・トラブル対応・データ移行](docs/operations.md)
- [開発・検証手順と検証範囲](docs/development.md)
- [構成と通信境界](docs/architecture.md)
- [公開時の安全性](SECURITY.md)
- [AIエージェント向け作業指針](AGENTS.md)

公開ドキュメントには認証値や実アカウントの調査情報を記載しません。

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

同じ検証済みの対象を継続して取得する場合、数値IDはGit管理外の `private/target-anchor-id`（0600）、セッションは `private/tiktok-sessionid.secret`（0600）に分けて保存し、次のスクリプトを使います。IDも認証情報として扱い、コマンド引数・シェル履歴・ログ・公開文書に値を書きません。スクリプトはこのチェックアウトのファイルだけを読み、`.env` やブラウザのログイン状態を自動取得しません。

```sh
bash scripts/replay-authorized.sh recordings
bash scripts/replay-authorized.sh download-all --workers 6 --max-gib 12
```

スクリプトは `TIKTOK_TARGET_ANCHOR_ID` と `TIKTOK_SESSIONID` を子プロセスの環境にだけ設定し、出力先 `downloads` と既定の `private/jobs/` を維持します。外部ディスクへ直接保存する場合は、マウントと保存先の作成・権限設定を確認してから `TIKTOK_DOWNLOAD_DIR=/Volumes/m.2-ssd/movie/tiktok bash scripts/replay-authorized.sh download-all` と実行します。`downloads` はそのディレクトリへのシンボリックリンクになります。既存の空でないディレクトリや別のリンクは上書きせず停止します。環境変数を未設定にしても既存のリンクはそのまま使い、ディスク未接続時は保存先を勝手に作成しません。通常のCLIでは従来どおり `download-all` の出力パスを直接指定できます。CLIの明示的な `--anchor-id` / `--user` は通常どおり使用できますが、スクリプトは誤った対象への切替を拒否します。録画一覧には対象ユーザーIDを表示しません（録画IDやタイトルは個人情報として扱ってください）。

対象指定はアクセス権を与える操作ではありません。**ログインしたアカウントが閲覧できる録画通知**を最後まで列挙し、対象の数値ユーザーIDと一致する実在の配信だけを取得します。通知にない配信・他アカウント限定録画は取得できません。

## YouTube の元動画（別経路）

視聴・保存する権利がある公開済み動画のみ、外部SSDに最高の利用可能な映像・音声ストリームを再エンコードせず保存します。Node.js 22+ と `brew install ffmpeg-full` が必要です。`uv.lock` の `youtube` オプションには yt-dlp、EJS 解析コンポーネント、ローカル日本語ASR用の mlx-whisper が含まれます。

```sh
bash scripts/download-youtube.sh 'https://www.youtube.com/watch?v=lL4opb4iB3w'
# 別の外部ディスクを使う場合:
YOUTUBE_DOWNLOAD_DIR=/Volumes/your-ssd/movie/youtube bash scripts/download-youtube.sh 'https://www.youtube.com/watch?v=lL4opb4iB3w'
```

保存先のディスクが接続されていない場合や既存の保存先が公開権限の場合は中止します。実体は外部ディスクに置き、ローカルの `youtube-originals` はシンボリックリンクです。原本は `YYYY-MM-DD_HH-mm-ss_JST_youtube-動画ID/同名.mp4` に保存します。公開時刻は `release_timestamp`（なければ秒精度の `timestamp`）をJSTに変換し、日付だけしか得られない動画は停止します。安定IDロックと既存ディレクトリ検査で再取得を拒否します。旧IDだけの原本も自動で再取得しません。既存の動画を上書き・削除しません。映像と音声は yt-dlp の `bv*+ba/b` による最良の利用可能な形式を選び、再圧縮せずMP4に結合・再格納します。AV1/VP9映像やOpus音声も元のコーデックを保持し、H.264/AACへの自動変換はしません。MKVへの出力やフォールバックは行わず、MP4に格納できない形式ならエラーで停止します。利用者のyt-dlp設定による出力形式の上書きも無効にします。再生互換性はプレイヤーとコーデックに依存します。供給形式・取得可否はYouTube側に依存します。取得したファイルは編集前の原本であり、字幕付与・BGM除去・無音カット済みの成果物ではありません。

新しいYouTube編集出力は `replay.youtube_delivery.new_edit_directory` で `公開日時JST_youtube-動画ID-edit-一意ID/` を原子的に予約し、そのディレクトリ名と同じstemでMP4・ASS・SRTを保存します。既存の編集原本・検証記録を自動削除する処理ではありません。公開時刻の根拠がないと編集ディレクトリも作成しません。

### 既存の字幕データからの再生成

`edits/youtube-<動画ID>/render-config.json`（`version`、`video_id`、元動画 `source_video`、
任意の音声 `source_audio`、`cut_frames`・`cues`（同ディレクトリからの相対パス）、`publication`、
`output_root`、`ffmpeg`、`ffprobe`）を置き、共通スクリプトで再生成します。

```sh
uv run --locked python -m replay.youtube_render edits/youtube-<動画ID>/render-config.json
```

出力は `<公開日時JST>_JST_youtube-<動画ID>-review/` で、TikTokのリプレイと同じく日付と`-review`を
含む名前です。字幕は白い太い縁取りの外側に細い黒縁を付け、話者名は表示しません。識別済みの話者が
2人以上いる場合は話者ごとに紫・緑・水色の単色で、1人の場合は水色、話者不明は灰色で
表示します。SRTに話者名を付けるのは識別済みの話者が2人以上いる場合だけです。

### TikTokと同じ字幕の表示切り替え

YouTube字幕もTikTokと同じ `make_captions` 方針で作ります。語が増えていき、行の文字数上限・文末・
無音・発話の区切りで新しい字幕に切り替わります（1語ずつ先頭が消える流れる表示にはしません）。
改行は句読点でのみ行い、語の途中では切りません。同じ話者の字幕が時間的に重なる場合は、同じ位置に
重ねず空いている最も下の段に積み上げます。

「〜よ」「〜かな?」のような2文字以内のひらがなの文末（同じ発話内で0.25秒以内に続き、直前が
句読点で終わらないもの）は、前の語と同じ字幕に残します。入りきらない場合はその前の語で先に
切り替え、末尾だけが単独で表示されることを避けます（TikTok・YouTube共通）。均等に折り返せる
YouTubeの描画では `tail_overrun` により文末分だけ文字数上限を超えることを許し、最終行が4文字
以下になる折り返しは行の長さを揃えます。

- 話者別の語トークンがある場合: `python -m replay.youtube_caption_rebuild CANDIDATE.json reviewed.csv out.csv --lane 1=left ...`
- 区間単位の確認済み字幕しかない場合: ローカルWhisperの語時刻で `python -m replay.youtube_caption_align reviewed.csv words.json DURATION out.csv`。
  確認済みの文字は変えず、認識語が文字の80%以上と一致した字幕だけを語の境界で分割します。
  一致しない字幕は元の表示のまま残し、文字数で時間を按分することはしません。

### 話者推定の照合

話者別に分離した字幕の話者を、話者分離（Nemotron等）と分離音声の音量で照合します。
話者分離が1人だけを字幕区間の80%以上で示し、最も大きい分離音声も同じ話者を示す場合に限り
話者を付け替えます。重なり・判定なし・`unassigned` は変更せず、話者を推測で作りません。

```sh
uv run --locked python -m replay.speaker_reconcile cues.csv diarization.json cut-frames.json out.csv \
  --lane 1=middle --lane 2=left --lane 3=right \
  --stem left=voice-1.wav --stem middle=voice-2.wav --stem right=voice-3.wav
```

### 設定ファイルによる自動字幕編集

動画ごとのJSON設定を共通スクリプトで処理します。実ファイルの設定は
`edits/youtube-<動画ID>/subtitle-config.json` 等のGit管理外に保存してください。

```sh
uv run --locked --no-dev --extra youtube python -m replay.youtube_subtitles CONFIG.json
```

設定項目は `version: 1`、`video_id`、`source`（入力動画）、`publication`
（検証済み公開時刻JSON）、`output_root`、`keep_bgm`、`audio_policy`、
`subtitle_mode: "burn_and_embed"`、`speakers`（1または3）、`model`
（ローカルWhisperモデル）、`weights_sha256`、`ffmpeg`、`ffprobe` です。
`keep_bgm: true` は `audio_policy: "original"` と原本入力を組み合わせ、
BGM・ゲーム音声を含む入力音声を保持します。`false` は
`"existing_reduced"` と既存のBGM低減済み入力を指定します。この字幕スクリプトは
新たな音声分離や無音カットを行いません。

字幕はASSから一度だけ動画へ焼き込み、同名SRTを保存します。焼き込み済み字幕と二重に表示されないよう、選択可能な字幕トラックは埋め込みません。編集済み動画（`youtube-edits/` 配下や `-edit-`・`-review` を含む名前）を字幕の入力にすると拒否します。
ASRはローカルの固定モデルだけを使用し、認識した区間時刻をそのまま使います。
単語時刻の補間や話者の推測割当は行いません。入力・既存編集を上書きせず、
一意な日付付き編集ディレクトリと認識・検証記録を作成します。
自動字幕の時刻・表記は推定値であり、話者混線の完全解消や誤認識ゼロを
保証するものではありません。モデル重みを配布したり録画を送信したりしません。

欠落・反復候補の追加認識も、動画別の設定から共通スクリプトで実行できます。

```sh
uv run --locked --no-dev --extra youtube python -m replay.youtube_subtitle_repair REPAIR_CONFIG.json
```

追加認識設定は `version: 1`、`source_config`（上記動画設定）、`source_job`
（設定が一致する既存編集ディレクトリ）、`output_name`、`evidence`（同編集内の監査記録）、
`windows` を指定します。各窓は入力動画の秒単位の `start`・`end`、対象区間の
`target_start`・`target_end`、`reason` を持ちます。原音の新しい文脈で認識し、
元の字幕・動画を変更せずに仮説と実音声時計を保存します。既存認識は再利用し、
設定の変更・不完全な音声・モデルの変更を拒否します。仮説の作成だけで
欠落が解決したとは判定しません。認識・照合後に修正版を別名で出力します。

照合済みの認識区間を選んで修正版を出力する場合は、次を実行します。

```sh
uv run --locked --no-dev --extra youtube python -m replay.youtube_subtitle_finalize FINAL_CONFIG.json
```

設定は `version: 1`、`source_config`、`source_job`、`review`（同編集内の照合記録）、
`selection` を持ちます。選択項目は `record`（同編集内の認識JSON）と
`segment`（元の区間インデックス）だけです。設定へ任意の字幕文や時刻を
書き込むことはできません。実入力WAVの時計・認識区間・原音設定を検証し、
同じ共通レンダラーでBGM方針と字幕焼き込みを維持した別名の動画を生成します。
MP4の埋込み字幕は同時表示する字幕の集合を元のSRT境界で区切り、
重なった字幕の末尾がコンテナーによって短縮されないようにします。
焼き込み・検索用SRTの認識区間や原文は変更しません。
ASR区間は推定値であり、選択・全編デコードの成功だけで表記や話者の正しさが
証明されるわけではありません。

Nemotron-3-Diarization の使用には、このモデルの GGUF（`pre_ln` 対応）を読めるビルドの `nemo-speech` が別途必要です。公開済みバイナリがモデルに対応しない場合、話者分離を成功したものとして扱わないでください。16kHzモノラルPCM WAVを `nemo-speech diarize input.wav --model /absolute/path/Nemotron-3-Diarization.gguf --preset offline --format json -o diarization.json` で解析できます。Nemotron-3-Diarization は音声上の話者時刻を推定しますが、話者番号と画面内の人物の対応や重なった声の文字起こしを自動で確定するモデルではありません。画面上の位置まで検証された `start,end,speaker,text` CSV（`speaker` は `left` / `middle` / `right`、判定不能な箇所は `unassigned` として灰色表示）に対して `uv run --locked python -m replay.speaker_subtitles cues.csv captions.ass --srt captions.srt` で、1920×1080向けの紫・緑・水色＋白縁取りのASSと、TranscriptViewer向けの重複発話を保持したSRTを作れます。検証できない語は `replay.diarized_words` が推測で話者に割り当てません。SRTと同名のMP4を編集ライブラリに配置した場合に限りTranscriptViewerが読み込みます。字幕焼き込みには libass 対応の `ffmpeg-full` が必要です。

### TikTok縦動画の字幕位置と色

**新しいMotion字幕の既定は採用案20です。** 値は
[`src/replay/caption_style_defaults.py`](src/replay/caption_style_defaults.py) に集約しています。
上色 `#2FB0EE`・下色 `#E3F6FF`、濃色を高さ75%まで保持し、残り25%で明色へ変化します。
白縁の参照幅は従来の0.8倍（36→28.8）、外側の黒縁は細さを保つため総幅34.8です。
内側の黒縁4・影は維持します。これらも同じ定数を使い、明示指定がある場合だけ別設定にします。
変更後のMotion校正・実描画は別途必要で、旧設定の動画を新設定の検証済みとは扱いません。
`replay.motion_finish` への接続と必要な校正入力は
[開発手順](docs/development.md#layered-motion-caption-finishing)を参照してください。
採用色は白とのAAA適合色ではなく、保存・候補XML生成だけで実描画の検証済みとはしません。
以下のASS描画・旧配色の記述は既存経路の説明であり、新しいMotion字幕の既定ではありません。

現在のMotion字幕経路は、元動画の正確なフレーム時計を持つVision観測から顔の位置と
縦横サイズを読み、頭上の概算位置を求めます。過去の大きい顔サイズや字幕位置を保持せず、
表示中のフレームの測定値を使います。左右中央・下揃えを維持し、微動だけを定数ファイルの
有界デッドバンドで抑制します。改行とページ切替は品詞・認識時刻・実描画寸法に基づきます。
観測がない区間は頭位置不明のまま、文字ブロックの実寸と画面内余白だけで上部に収める
`caption-only-screen-layout`候補とします。過去の顔や校正位置へ戻す処理は使いません。
この候補は顔回避・頭近接の合格を意味せず、観測復帰時は現在の顔に基づく配置へ戻ります。

字幕の色は単独話者のYouTube字幕と同じ、青寄りの水色系（#005A9C）・太い白縁・外側の細い黒縁です。
指定 sRGB 色と白縁のコントラスト比は 7.1367:1 です。実出力の色・縁取りの検証や映像全体の AAA 適合を保証する値ではありません。黒い背景帯は使用しません。
改行は句読点の後で行い、1行18文字を超える場合は行の長さを揃えて分けます。候補字幕
（未レビュー）は認識結果の粗さにかかわらず、すべて同じ2行・36文字の表示枠で区切ります。
ネイティブ描画では位置・色を個別に指定できないため、各パートの既存の再エンコード時に
libass（`ffmpeg-full`）でASSを焼き込みます。完了判定は、計画に記録したASSのSHA-256と
一致する場合だけ字幕焼き込みを認めます。

字幕の文字サイズは、従来のネイティブ描画（56）とほぼ同じ字面になるよう実測したlibassの66で、
1行18文字と縁取りが画面幅に収まります。縁取りはYouTube単独話者の比率（白5・黒2を文字52に対して）を
この文字サイズに合わせた白6・黒3です。

### 再生成の処理時間

字幕だけを作り直す場合、BGM除去は前回と音声パケットが同一（MD5一致）なら同じ固定モデルで分離済みの
音声をそのまま使い、分離と待ち時間を省きます（`separation_reused_from` に記録）。各パートは
元字幕の検出領域を局所補修した素材を用意し、`python -m replay.part_render` によるFFmpeg 1回の処理で、
切り出し・30fps化・字幕の焼き込み・エンコードを行います。固定矩形ぼかしの生成経路は削除されており、
旧ぼかし指定を含む計画は拒否します。素材の局所補修と品質検証は別工程で必須です。パートは1配信あたり3つずつ並列に
描画し（`RENDER_PARALLEL_PARTS`）、元字幕のOCR（`OCR_PARALLEL`、既定3）と音声分離
（`SEPARATION_PARALLEL`、既定2）も独立したまとまりごとに並列実行します。時刻検証用のプロキシ動画は
削除せずに再利用します。
依頼による作り直しは `python -m replay.edit_completion regenerate ROOT --reason 理由` で旧完成版を
置き換え対象として記録し、新しい完成版が合格してから旧版を削除します。

### 文字起こしの欠落区間の再認識

長尺のWhisper認識は30秒の窓ごと読み飛ばすことがあります。認識後、分離音声が明確に
鳴っている（録音自身の雑音床+18dB以上）のに単語が1つもない2秒以上の区間を検出し、
同じ固定モデル・同じ設定でその区間だけを再認識します。既存の単語の時刻は変更せず、
前後の認識済み単語の間に中点がある単語だけを追加し、区間と結果を
`transcripts/full-mlx/gap-recovery.json` に記録します。全体認識の結果は
`raw-transcript.candidate.json` にそのまま残します。

### YouTube字幕の欠落区間の補完

`python -m replay.youtube_caption_recovery cues.csv vocals-16k.wav diarization.json cut-frames.json out.csv provenance.json --lane ID=lane --stem lane=voice.wav`
は、カット後の時間軸に並べた分離ボーカルで声がはっきり鳴っているのに字幕がない2秒以上の
区間を、固定モデルで再認識して字幕を追加します（区間内に収まる語だけを採用し、既存の字幕は
変更しません）。話者は話者分離と最も大きい分離音声が一致した場合だけ割り当て、一致しない
場合は話者不明（灰色・中央）にします。認識器は認識ランタイムから渡します。複数話者の字幕は
単独話者と同じ白縁・黒縁で、紫・緑・水色の単色にしています。

### YouTube編集動画の再生成

`python -m replay.youtube_render CONFIG.json --replace` は隠しディレクトリで描画・全編
デコード検証を終えてから日付付き出力と入れ替えます。描画中も旧動画はそのまま再生でき、
書き出し途中の動画をTranscriptViewerが開くことはありません。TranscriptViewerは動画を
開けない場合、字幕の問題ではなく動画の問題として表示します。

## 重複防止と互換性

- 安定した配信IDで重複判定。完了済み・他プロセスが取得中の配信は再取得しません。
- `private/jobs/` が完了記録です。削除しないでください。既存の出力や不整合を検出した場合、勝手に上書きせず検証を要求します。
- `download-all` の新規保存先は `downloads/YYYY-MM-DD_HH-mm-ss_JST_replay-配信ID.mp4`。APIの配信開始時刻 `start_time` を日本標準時（UTC+09:00）に変換するため、ファイル名で配信日時順にソートできます。同じ秒の配信もIDで区別します。
- 通知日時、ファイル作成日時、APIの `create_time` から開始時刻を推測しません。新規取得で開始時刻が不明・不正な場合は停止します。署名付きURL・認証値は一覧に出力しません。
- 旧名の保存済みファイルもIDで重複判定します。改名する場合はファイルと完了記録を一緒に更新してください（[改名手順](docs/operations.md#rename-completed-recordings)）。低水準の `download` コマンドは明示的に指定された出力パスを使います。
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

## TranscriptViewer

`bash TranscriptViewer/build-app.sh` でmacOSアプリをビルドします。動画と対応するCSV/SRTを
読み込み、既存ライブラリにも次の機能を利用できます。

- 再生速度を **1.0x / 1.5x / 2.0x** から選択。一時停止・再開、動画切替、時刻ジャンプでも選択速度を維持します。
- 「要約」タブで **Apple Intelligence（Foundation Models）** が動画全体と10分ごとの
  内容を自然な日本語の文章にまとめます。発話の抜粋や箇条書きではなく、主題・話題の流れ・
  明示された理由や結論を説明する段落形式です。字幕の累積表示だけを整理し、全発話を
  処理対象にします。長い入力は分割して中間要約を作り、区間要約から全体要約を生成します。
  認識誤り・AIの誤り・情報の取りこぼしは残るため、必要に応じて動画で確認してください。
  文字起こしのある区間だけ表示し、境界をまたぐ発話は開始時刻の区間に含めます。
  区間時刻をクリックすると、その10分区間の先頭から再生します。
- 左の動画一覧を右クリック → **Reveal Finder** で、その動画をFinder内で選択表示します。

AI要約には **macOS 26以降・Apple Intelligence対応Mac・Apple Intelligenceの有効化と
日本語対応モデルの準備完了**が必要です。利用できない理由や生成失敗は画面に表示し、
発話の抜粋で代用しません。OS設定やモデルのダウンロードをアプリが勝手に変更することはありません。

要約タブで動画を選ぶと自動生成し、進捗を表示します。動画やタブの切替で生成を中断し、
古い動画の結果を新しい動画に表示しません。左カラム下部の「新しい順にすべて要約」は、読み込み済み動画だけを
配信日時の降順（日時不明は最後）で順次処理します。一括生成は動画・タブの切替でも続き、
成功・失敗件数を別々に表示します。停止後や再実行時は完成済みを再利用します。
箇条書き形式の応答には、元の資料を処理して得たAIの下書きを、AIで文章に整える処理を
一度試します。記号を消して文章に見せかけることはせず、直らなければ失敗を明示します。
Appleの既存テキスト変換用設定を使用し、明示的な辞退文は要約として表示しません。
完成した要約は字幕の隣の `<字幕ファイル名>.summaries/` に保存し、再起動後も再利用します。
字幕から導出した入力のSHA-256・処理バージョン・全区間の時計・文章形式が一致する場合だけ読み込みます。
字幕が変わった場合や「再生成」を押した場合は作り直します。以前の完成版は履歴として保持し、
途中失敗で完成版を上書きしません。文字起こしを外部送信せず、動画・字幕ファイルにも書き込みません。
要約と認識結果はタブで切り替えます。

### 要約CLIと調査

CLIとアプリは `TranscriptSummaries` の生成サービスと `TranscriptCore` の要約処理を共用します。
CLI専用の別プロンプト、別モデル、抜粋へのフォールバックはありません。

```sh
# リポジトリのルートでビルド
swift build --package-path TranscriptViewer -c release --product transcript-summary

# 読み込める既存動画を新しい順に生成。完成済みは再利用
TranscriptViewer/.build/release/transcript-summary generate --library edits

# 最新1本を全文処理し、詳細な調査記録をローカルに保存
TranscriptViewer/.build/release/transcript-summary generate --library edits --limit 1 \
  --diagnostics edits/work/summary-debug --include-text

# モデルを呼ばず、保存済み要約の整合性を検査
TranscriptViewer/.build/release/transcript-summary inspect --library edits
```

`--transcript PATH` でカタログ内の字幕を1本指定できます。`--limit` は動画数の上限であり、
文字起こしを途中で切り捨てません。`--force` は旧版を保持した再生成です。
アプリで「保存済みを読み込む」を押すと、CLIの結果を再推論なしで読み込みます。
通常の動画選択でも保存済みを優先します。同じ字幕の同時生成はプロセス間ロックで拒否します。

編集動画の完成前には、原本時計であることを由来記録で確認した ASR の
`labels`（`start`・`end`・`text`）JSON を直接指定できます。
`transcript-summary generate --source-labels ABS_JSON --source-media ABS_ORIGINAL
--clock-domain source-presentation` は時刻を変換せず同じ要約処理を実行し、
JSON の隣の `.summaries/` に保存します。`inspect` も同じ引数で利用できます。
原本媒体は同一性の参照のみでデコードせず、MKV にも対応します。
これは Viewer への編集動画登録ではありません。後の字幕と時計・文章が変われば、
その最終字幕を入力として要約を検証・生成してください。
生成中は `.summaries` ディレクトリや `.generation.lock` を移動・削除しないでください。

診断ディレクトリは実行ごとに分かれ、入力の対応表、断片／区間／全体の処理種別、区間開始秒、
校正の有無、入力バイト数、応答トークン予算、所要時間、応答サイズ、エラーコードをJSONに残します。
要求の直前にも記録するため、応答が戻らなかった箇所も確認できます。
`--include-text` を指定した場合だけ指示・プロンプト・生のモデル応答を含めます。
**要約・診断は個人データです。Gitや公開ログに載せないでください。** ディレクトリ0700・ファイル0600で保存します。
CLIの終了値は全件利用可能=0、一部失敗／未生成=1、引数・初期化エラー=2です。
成功終了は文章の意味的正確性を保証しません。失敗時は診断を確認し、未完成の要約で代用しません。

### 要約のWebビューア公開

`transcript-summary generate` が字幕の隣に保存した要約を、非公開のWebビューア
（Cloudflare Worker、X認証）へ反映するには次を使います。要約の生成・編集は行わず、
保存済みの検証済み要約（`current.json`）だけを、ビューアの `data/<id>.json` と
`catalog.json` に写し、変更分だけを `cf r2 objects put` でR2へ送ります。
要約が保存されていない動画には触れません。

```sh
# 反映内容だけ確認（R2へは送らない）
uv run python -m replay.viewer_summary_publish --web-dir edits/work/cloudflare-viewer \
  --bucket transcript-viewer-media --dry-run
# 反映してアップロード（`cf` にログイン済みであること）
uv run python -m replay.viewer_summary_publish --web-dir edits/work/cloudflare-viewer \
  --bucket transcript-viewer-media
```

## 作業ファイルの保存規約

**作業はこのリポジトリ内だけで行います。`/tmp`・`/private/tmp`・OSの一時ディレクトリは、
Markdownを含めて使用禁止です。** 引き継ぎ文書、調査メモ、スクリプト、ログ、取得途中の
ファイル、テスト成果物はGit管理外の `edits/work/`、モデルは `edits/models/` に保存します。
外部を指すシンボリックリンクを作業先に使わず、実体がリポジトリ内にあることを確認してください。
子プロセスの `TMPDIR`・`TMP`・`TEMP`、テストの一時領域、ビルドキャッシュ、ログ出力先も
リポジトリ内に設定します。外部一時ディレクトリへの出力を強制するツールは使用しません。
このMacではFoundationの `temporaryDirectory` / `NSTemporaryDirectory()` が `TMPDIR` を
無視することをコンパイル済みの小さな検査プログラムでも確認しました。環境変数の設定だけで
リポジトリ内になると仮定せず、実際のパスを確認し、必要な処理では保存先を明示的に渡します。
これは今後の作業規約であり、既存実装すべてが自動的にこの配置を強制するという意味ではありません。
既存の原媒体・外部保存先・完了記録・過去成果を勝手に移動、削除、上書きする許可ではありません。

<a id="local-models"></a>

## ローカルモデル：形式・固定版・検証状況

モデル名だけで互換性を判断しないでください。通常の録画取得にASR・音声分離モデルは不要です。
以下はローカル編集とネイティブ実装の検証用で、モデル本体や実録画はGit管理しません。

| 用途 | 形式・保存先の規約 | 状態 |
| --- | --- | --- |
| 使用する字幕ASR | MLX、`edits/models/mlx-whisper-large-v3/49e6aa286ad60c14352c404340ded53710378a11/` | `replay.local_asr` の既定モデル。指定された非Turboのlarge-v3。取得・実推論の検証は下記参照。 |
| 旧Python ASR | MLX、`edits/models/mlx-whisper-large-v3-turbo/model/` | 旧成果の再現用に保全。新しい字幕ASRの既定には使わない。 |
| WhisperKit ASR（別経路） | コンパイル済みCore ML、`edits/models/whisperkit-coreml/<revision>/<model>/` | 全22ファイルを取得・ハッシュ検証済み。合成音声の推論成功。全nativeゲートは未合格で、今回の字幕ASRには使わない。 |
| Whisper tokenizer | ローカルJSON、`edits/models/whisper-tokenizer/<revision>/` | 既存資産で日本語往復変換テスト成功。リポジトリ内にも3ファイルを取得・ハッシュ検証済み。 |
| 音声分離 | Core ML、`edits/models/htdemucs-coreml/compiled/HTDemucs_SourceSeparation_F32.mlmodelc` | 既存native検証で使用。ファイル名のF32に反し実入出力はFloat16なので実契約を検証する。 |
| アバター顔・元字幕の検出 | PyTorch（Ultralytics YOLO11n）、`edits/models/yolo/avatar-face-caption-v2.pt`（sha256 `8c6e45b4b81cc4cc550ff988cbd4ab800c8cbd4db82dcfff5bfa3a46dda6b488`、`.sha256` 併置）。初期重みは公式 `yolo11n.pt` を1回だけ取得して固定。旧 `avatar-face-caption-v1.pt`（sha256 `f486fd49…`）は旧成果の再現用に保全。 | MPS・640pxで約3.3ms/フレーム。6収録で学習し、学習に未使用の10-07・10-12・10-14aで字幕画素被覆99.8〜99.9%、顔検出100%。字幕削除（YOLO admission＋修復マスク）と固定Y配置（`yolo_track` v2＋`caption_fixed_y_job`）で本番利用。既存の品質ゲートは不変。使い方・再学習・新収録の受け入れ確認は [docs/yolo-detector.md](docs/yolo-detector.md)。 |

### 使用するMLX Whisper large-v3（非Turbo）

ユーザー指定の [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx)
（MIT）を既存のPython/MLX経路で使用します。CoreML/Neural Engine向けコンパイルは行いません。

- 固定revision：`49e6aa286ad60c14352c404340ded53710378a11`。
- 重み：`weights.npz`、3,083,520,416 bytes。
- 重みSHA-256：`05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d`。
- `config.json` SHA-256：`34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3`。
- 保存先：上表のrevision付きローカルディレクトリ。配布4ファイルの総容量は3,083,522,487 bytes。
- 取得・ハッシュ照合・合成音声での実推論の記録：`edits/work/mlx-large-v3-switch/`。
  配布4ファイルのサイズ・ハッシュ照合済み。既定CLIで2.9秒の合成日本語音声を約7秒で処理し、
  9語グループ・未解決のゼロ長語0、入力保全を確認しました。実録画の品質検証とは別です。
  設定はencoder/decoder各32層・128 mel bins・語彙51,866です。部分ファイルは利用しません。

インストール済みのモデルとロック済み `youtube` extra のMLXランタイムを使い、
リポジトリのルートから次の既存CLIで文字起こしします。

```sh
uv run --locked --extra youtube python -m replay.local_asr INPUT.wav NEW_EVIDENCE_DIR
```

モデルパスと設定・重みのハッシュは上記固定版が既定です。取得は推論と分離し、未配置・不一致なら
停止します。入力は60秒以内の16kHz・モノラルPCM16 WAVで、日本語の語時刻を保存します。
既存成果は上書きせず、認識は未レビュー扱いです。元動画の時刻への投影と字幕品質検証は別途必要です。
明示的な `--model` / `--weights-sha256` / `--config-sha256` による旧モデルの再現機能は残します。

長尺の準備済みモノラルPCM16・16kHz WAVには、同じ固定版モデルの共有バッチ処理を使えます。

```sh
uv run --locked --extra youtube python -m replay.batch_asr INPUT.wav NEW_EVIDENCE_DIR
```

最大6時間の入力を60秒窓・30秒間隔で処理し、同一プロセスでモデルを再利用します。
入力・モデル・実装のハッシュと窓ごとの実サンプル範囲、未加工の認識結果を保存します。
同じコマンドの再実行では検証済み窓だけを再利用し、変更・欠損・途中書き込みは上書きせず停止します。
空の認識窓も保持しますが、無音とは判定しません。完了記録は生のASR収集の完了であり、
重複窓の統合、語時刻の正しさ、元媒体の時刻への投影、字幕表示や編集済み動画の完成を意味しません。
音声の正規化・リサンプリングは事前に行い、その時刻対応を別途保持してください。

完了したバッチは `python -m replay.asr_assembly EVIDENCE_DIR NEW_CANDIDATE.json` で
入力・窓のハッシュを照合し、重複窓の中点所有規則で候補語を選択できます。
ゼロ長語・重なり・時刻逆転は保持・計数し、時刻を補間したり並べ替えたりしません。
これは入力PCM時計上の候補であり、元動画時計への投影・語の正確性・字幕完成の承認ではありません。

### WhisperKit向けに特定した候補（別経路・履歴）

- 正規配布元：[argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml)（MIT）。
- 固定revision：`0f63a7800b00dd0226abd051b906c246e1907482`。
- サブディレクトリ：`openai_whisper-large-v3-v20240930_turbo_632MB`。
- 公開ファイル一覧の合計：**22ファイル、645,668,913 bytes**。名称の632MBから容量を推測しません。
- `config.json`：decoder layers **4**、mel bins **128**、vocabulary **51,866**。
- TextDecoder：key/value cache更新形状 `[1,5120,1,1]`、語時刻用の
  `alignment_heads_weights` 出力 `[1,1500]`。
- 必須ディレクトリは `AudioEncoder.mlmodelc`、`MelSpectrogram.mlmodelc`、
  `TextDecoder.mlmodelc`。配布には `TextDecoderContextPrefill.mlmodelc`、
  `config.json`、`generation_config.json` も含まれます。
- **日付のない** `openai_whisper-large-v3_turbo_954MB` は別の候補です。
  直接確認した内部構成は32層・cache更新 `[1,40960,1,1]` でした。
  `turbo` という接尾辞だけで4層版と同一視しないでください。

[固定版ファイル一覧](https://huggingface.co/api/models/argmaxinc/whisperkit-coreml/tree/0f63a7800b00dd0226abd051b906c246e1907482/openai_whisper-large-v3-v20240930_turbo_632MB?recursive=true&limit=100)
を取得記録として保存し、全件のパス・サイズと、LFSならSHA-256、通常ファイルならGit blob
SHA-1を照合します。Git blob SHA-1はファイル内容だけのSHA-1とは異なります。
`.part` は検証成功後だけ通常名へ移し、既存ファイルや失敗記録を上書きしません。

### トークナイザーとテスト設定

対応候補は [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) の
revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` です。
`config.json`・`tokenizer.json`・`tokenizer_config.json` の3ファイルを検証して保存します。
HFキャッシュのリンク先がディレクトリ外にある場合、そのままではnative側のローカル境界検証に
適合しません。境界検証を弱めず、承認された資産を通常ファイルとしてリポジトリ内の専用
ディレクトリへ保存・照合してください。推論中のHub取得やMLXへの自動切替は行いません。

nativeテストは子プロセス環境の `WHISPER_TEST_MODEL`、`WHISPER_TEST_TOKENIZER`、
`DEMUCS_TEST_MODEL` で検証済みのディレクトリを明示します。変数の設定だけではモデルの
配置・完全取得・実推論成功を証明しません。上記保存先は規約であり、自動インストールされません。

**取得前の検証結果：** tokenizer単独2テストは成功し、指定後のnative全379テストは
377成功・Core ML本体未設定の2失敗でした。今回の全TSan試験では異常は再現しませんでしたが、
先行試験のSEGVは原因未確定です。Core MLの初回取得は `curl` の
`Maximum file size exceeded`（exit 56）で停止しました。リポジトリ内のHTTP対照検査で、
243-byte本体を受信する前の302応答で停止していたことを確認しました。
HEADで転送先を解決し、同じ243-byte上限の直接GETを行うと、実サイズ・SHA-256が一致しました。
転送先は正規配布ドメインに限定し、署名付きURLをログ・引数に残さず、TLS検証と各ファイルの
厳密な容量上限を維持します。この方式で、Core ML全22ファイルとtokenizer全3ファイルを
上記リポジトリ内の保存規約どおりに取得し、固定版manifestとのサイズ・ハッシュ一致を確認しました。
検証記録の場所は `edits/work/whisper-investigation/fixture-transfer-path.txt` に記載しています。
実推論と全品質ゲートは、元の開発チェックアウトを変更せず、リポジトリ内の検証用ソースコピーで
改めて確認します。取得成功を実推論成功や先行TSan異常の解消とは扱いません。
モデルの取得、実推論、全品質ゲート、動画の字幕修正・実描画検証は別々に判定します。
引き継ぎは `edits/work/whisper-investigation/HANDOFF.md` に保存します。

**後続の限定検証：** 固定版Core MLモデルの合成日本語推論と単体CLIのJSON確認、
リポジトリ内での逐次focusedテストは成功しています。tokenizerの3ファイルはハッシュ検証済みで、
日本語往復変換も確認済みです。ただし、先行する並行CLIのタイムアウトを解決済みとはせず、
全nativeゲートは未検証のままです。モデル名・固定版・検証状況はこの節に集約し、
AGENTSやskillに別の固定版一覧を複製しません。

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
