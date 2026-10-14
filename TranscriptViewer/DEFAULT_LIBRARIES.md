# Default additional libraries

Ordinary startup keeps the saved library (or the checkout's `edits` library) and
adds the libraries configured in `edits/viewer-default-libraries.json`. This is
local data, excluded from Git. For example:

```json
{"version":1,"libraries":["/absolute/path/to/edited-deliveries"]}
```

The list is limited to 16 distinct absolute, normalized directory paths. Linked
configuration or library paths, `private` and `downloads` are rejected. A volume
that is temporarily absent remains configured and produces a visible scan issue.
An invalid configuration leaves the primary library available and reports the
problem. Explicit `--library`, Open Document and folder selection remain scoped
to that chosen library and do not change the default configuration.

The default view rereads configuration and candidate selections every 30 seconds,
on activation and with Refresh. A newly published movie appears only after the
existing catalog's candidate, media identity, source date and SRT checks pass.
A current candidate replaces older entries of the same recording and date;
contradictory dates are reported instead of merged. Unrelated videos remain.
Refresh retains the current player and position while its selected movie remains.
Candidate and summary labels retain their verification status; loading is not
quality or synchronization approval. This configuration never writes media,
transcripts, summaries, candidate records or saved folder preferences.

Verification uses synthetic `DefaultLibraryManifestTests`,
`DefaultCatalogSessionTests`, `CatalogScanMergeTests`, and the existing startup,
catalog, playback and summary suites. Run `make swift-check` and `make swift-tsan`
with repository-local scratch. Actual default startup and playback are separate
runtime checks.
