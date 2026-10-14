# Agent working guide

Before changing behavior, read [README.md](README.md), [SECURITY.md](SECURITY.md) and [architecture](docs/architecture.md). Use [development](docs/development.md) for verification and [operations](docs/operations.md) for troubleshooting. Detailed procedures are loaded through the task routing below; skills do not grant permissions or prove completion.

## Scope and safety

- This macOS CLI handles authorized, observed, finished TikTok LIVE replays. It is not a screen recorder or access-control bypass. Never weaken TLS, authenticated endpoints, media-host scope, byte limits or account-continuity checks; do not guess room IDs or alter signed URLs.
- Work only inside this checkout. No `/tmp`, `/private/tmp`, OS temporary directories or other out-of-repository scratch, including Markdown, scripts, logs, downloads and test artifacts. Use ignored `edits/work/` for work and `edits/models/` for models; reject escaping symlinks and verify actual child-process scratch/cache/log paths. Environment variables alone are not proof; see `repo-jobs`.
- Treat `private/` and `downloads/` as user data: do not recursively read, index, attach, stage, clean or delete them during coding. Never expose credentials in prompts, output, logs, examples, tests or command arguments; disable shell tracing and HTTP debug logging.
- For acquisition, check this checkout's `private/target-anchor-id` and `private/tiktok-sessionid.secret` without printing values; load them through `scripts/replay-authorized.sh`. Do not infer creators from replay IDs/jobs, substitute a home `.env`, or assume automatic `.env` loading. Use an explicitly supplied project `.env` only when needed.
- Do not delete completion records or bypass deduplication to force acquisition. Missing/changed completed output requires investigation, not redownload. Retention and cleanup follow the disk-space policy below.
- Routine tests must not launch login windows, reuse browser profiles, manipulate desktop input/focus or operate connected devices. Login, OS permissions and purchases need authorization; never solve/bypass security challenges or fabricate review.

## Disk-space policy: 空き容量と保存・削除

- **空き容量は重要資源。** 大容量処理前に出力先の空き容量と、入力・中間・最終出力・並行ジョブを含むピーク追加使用量を見積もる。起動ボリュームの通常下限は **30GiBまたは総容量の5%の大きい方**。下限を維持できない新規大容量処理は開始しない。工程ごとに再測定し、不足時は承認済みの整理と容量を増やさない作業を優先する。
- **保存**：原本、完成品、モデル、保存プロジェクト、採用中のパート、再開に必要な部分成果、字幕・要約、`private/jobs/`、設定・manifest・ハッシュ・検証結果・失敗原因の記録。検証や障害再現の唯一の根拠も保持する。
- **削除**：不要と確認できた中間合成・一時mux・旧字幕alpha・キャッシュ。過去成果という理由だけで無期限に保持しない。**承認済みなら再承認を求めず、安全確認できた対象から削除する。** 全動画の完成を一律に待たず、依存不明の対象だけ保留する。
- 削除には現行・待機・再開予定ジョブ、完了判定、FCP/Viewerの参照確認が必要。置換による削除は代替成果を検証する。古さ・大きさ・未オープンだけで判断しない。正確なパスと根拠を記録し、直前に同一性・使用状況を再確認する。手順は [disk-maintenance](.agents/skills/disk-maintenance/SKILL.md) に集約する。
- 共有ディレクトリの一括削除、稼働中ファイルの移動、他エージェントやアプリの中断はしない。リポジトリ外へ操作範囲を広げない。論理削除量と実測空き容量増加を区別し、ハードリンク/APFS共有領域を二重計上しない。この下限はハング防止の保証でも原因診断でもない。

## Execution and completion

- **ユーザーの追加指示を優先する。** 最新の直接指示を旧計画・同じ優先順位の過去指示・エージェントのメモや監視方針より優先し、矛盾する部分を直ちに更新する。変更されていない要件は維持する。追加指示を単に文書へ引用して終わらせず、対象・実作業・合格条件・実施結果に結び付ける。「削除」を「使用しない」で代替せず、未実施は未完了として残す。特定済みの対象や承認を繰り返し要求しない。不明点はまず既存会話・証拠で確認する。実際の上位指示との衝突だけは原文・出典・対象操作を明示して切り分け、自己作成メモを上位ルールに昇格させない。[追加指示の反映手順](docs/blocker-audit.md#latest-user-requests)を使う。
- **完成物を優先する。** 最終出力までの入出力・合格条件・依存を先に確認し、代表的な短区間で実処理を接続する。まず1本を全尺で完成させ、共通処理を残りへ展開する。残りの依頼を省略せず、短区間の成功を全尺の完成としない。
- Continue independent safe, authorized work without repeated confirmation. Isolate blocked stages and record the exact prerequisite. Set a hypothesis, minimal input, success/failure criterion and time bound for diagnosis; fix the shared cause, test, then reconnect processing. Stop retries that add neither evidence nor progress. Work on downstream integration with synthetic inputs while waiting; never treat unverified media as approved.
- Before declaring a whole goal blocked or ending its loop for input, use the [blocker audit](docs/blocker-audit.md). Trace a restriction to its original author, scope and recheck condition; an agent-authored continuation or helper limit is not a new user prohibition. Unknown prerequisites need bounded local checks, not assumed human permission. Review independent work first; never auto-remove a real restriction. This repository check does not intercept external goal/loop tools.
- Use the existing `replay.workflow` dependency preflight (`--check`) before multi-stage execution; declare producer outputs, consumer inputs and domain verifiers. Use `scripts/tmux-job.sh` for detached local commands and inspect its retained launch/log/status evidence before retrying. Keep stable purpose-named work roots, not random task roots or copied runners. A valid graph or exit zero is not delivery acceptance.
- Reuse results whose input/code/model hashes and completion records match. Re-run only affected stages, avoid duplicate jobs and unnecessary copies/re-encoding, and update estimates from measured throughput, remaining work and contention. Partial-stage estimates are not delivery promises. Inspect uncertain job outcomes before retrying; a timeout is not completion.
- Task-related app restarts are pre-authorized, including force termination after failed normal termination or an unresponsive app. Preserve recoverable work; unsaved state may be lost. Verify exact target, termination, restart and readiness, inspect existing jobs/outputs, record recovery and resume. This grants no cleanup, unrelated app/OS restart, login, purchase, permission or security-bypass authorization; details are in `repo-jobs`.
- Report completed/requested videos, intermediate progress, blockers and next actions separately. Distinguish partial tests, candidates, encoding, full decode and fully verified delivery; disclose unfinished requested summaries/Viewer integration. Never skip required stages, weaken gates, hide failures behind footage-specific settings or invent timestamps/review to claim completion.

## Task routing and ownership

| Task | Read / use |
| --- | --- |
| User addition/correction, superseded plan or missing requested action | [latest user requests](docs/blocker-audit.md#latest-user-requests); version-2 audit with `--latest-request` |
| Whole-goal blocker, input-wait stop or inherited restriction | [blocker audit](docs/blocker-audit.md) and `python -m replay.blocker_audit SNAPSHOT.json --require-blocked` |
| Job launch, monitoring, scratch, retry or recovery | [repo-jobs](.agents/skills/repo-jobs/SKILL.md); reuse existing shared scripts |
| Capacity planning or approved intermediate cleanup | [disk-maintenance](.agents/skills/disk-maintenance/SKILL.md) |
| Repository replay processing, captions, models, FCP/Viewer and delivery | [replay-apple-workflow](.agents/skills/replay-apple-workflow/SKILL.md) |
| Direct Apple app operation | Generic [apple-pro-apps](.agents/skills/apple-pro-apps/SKILL.md) and the relevant sibling `apple-*` skill |
| Source/test ownership and implementation boundaries | [architecture](docs/architecture.md#where-to-make-changes) |
| Verification and publication | [development](docs/development.md#quality-gate), [SECURITY.md](SECURITY.md) |

Project skills depend on generic skills, never the reverse. Do not embed replay paths/IDs, models, style or delivery policy in generic Apple skills. Reuse existing implementations/scripts rather than copying them into a skill. Skills are instructions, not enforced safety gates.

## Implementation invariants

- Use the configured Python patch version and `uv.lock`; no alternate environment manager. Install development dependencies with `uv sync --locked --extra capture --extra web`. Keep public APIs typed, run strict mypy and validate external JSON; avoid unchecked `Any`, casts, mutable defaults and broad exception handling that hides failure.
- Keep API authentication separate from profile lookup/CDN transfer; ordinary listing, acquisition and session reissuance must remain Android-independent. Preserve atomic private writes, fingerprint-bound metadata, stable-ID locks, non-overwriting publication and QuickTime compatibility. Automatic filenames require validated API `start_time` in JST plus stable ID, never inferred dates; authorized renames must update completion records without redownload/overwrite.
- **動画固有の処理は最小限にし、動画ごとの最適化は汎用処理で行う。** Pythonは `src/replay/`、nativeは `native/apple-pro-apps/` に実装する。動画固有の情報は入力・検証済み計測値・明示的な編集意図・設定・結果に限定し、共通アルゴリズムが映像・音声・字幕・処理負荷の実測からパラメータを導出する。ID・日付・特定尺/フレーム数を条件にした分岐、専用スクリプト、根拠のない固定値や例外設定で問題を隠さない。問題は共有実装で修正し、異なる特性の合成入力で回帰確認する。過去snapshotを処理backendにせず、観測を捏造しない。詳細は `replay-apple-workflow`、証拠の保存は容量管理規約に従う。
- Default captions use **style 20** from `src/replay/caption_style_defaults.py`; ASR uses **`mlx-community/whisper-large-v3-mlx` (not turbo)**. Do not silently substitute or auto-download models during inference. Exact model pins/status live in [README — local models](README.md#local-models); style calibration lives in [development](docs/development.md#layered-motion-caption-finishing).
- Every edited replay requires `edits/replay-<id>/edit-config.json` and `python -m replay.edit_completion check`: music removal, non-speech cuts, source-caption masks, transcription and burned dynamic captions must pass. No per-replay fallback, skipped/degraded stage or partial verification may be presented as finished delivery.

## Verification and Git handoff

- Add focused regression tests with synthetic inputs/mocked network and local FFmpeg fixtures where appropriate; never use private recordings as fixtures. Run **`make check`**, **`uv lock --check`**, **`uv build`**. Keep format/lint/strict types and at least **90% statement coverage per production file**; do not add skips/suppressions, lower coverage, or disable hooks/signing to obtain a pass. Native verification follows its README and development guide.
- Inspect existing changes and preserve unrelated edits. Do not commit, push, rewrite history or delete backups without authorization. Before publication, review/stage only intended paths and run **`make public-check`** (index and referenced history, not arbitrary unstaged changes). Keep publication/sdist allowlists narrow; review author/committer emails separately from secret scanning.
- Report exactly what was tested, remaining limits and whether test media was removed. Unit/FFmpeg tests, real login/listing, partial receipt, complete downloads, full decode and app playback are distinct evidence; a reissued cookie does not prove lifetime extension. Use [evidence levels](docs/development.md#evidence-levels).
