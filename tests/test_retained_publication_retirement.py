"""A terminal pending child may retire only its proven publication alias."""

import json
import os
from pathlib import Path

import pytest
from test_recovery_source_finalization import Case
from test_recovery_source_finalization import case as case
from test_retained_cleanup_finalization import ready_case as ready_case
from test_retained_publication_support import published_case as published_case

from replay import leaf_retirement as leaves
from replay import verified_hardlink_publication as publication
from replay.fcp_part_assembly import _reference
from replay.stages import file_digest


def test_real_pending_context_alias_retirement(
    published_case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = published_case
    monkeypatch.setattr(publication, "ROOT", Path.cwd())
    monkeypatch.setattr(publication, "_actors", lambda references: [])
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: set())
    keeper = c.output / "clean-source.mov"
    alias = c.output / "pending.mov"
    os.link(keeper, alias)
    info = alias.stat()
    inventory = c.output / "alias-inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "path": str(alias),
                        "kind": "file",
                        "classification": "hold",
                        "dev": info.st_dev,
                        "inode": info.st_ino,
                        "nlink": info.st_nlink,
                        "bytes": info.st_size,
                        "mtimeNs": info.st_mtime_ns,
                        "ctimeNs": info.st_ctime_ns,
                        "linkTarget": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    approval = c.output / "approval.md"
    approval.write_text("Synthetic owner approves only this redundant alias.", encoding="utf-8")
    terminal = c.output / "terminal.status"
    terminal.write_text("1\n", encoding="utf-8")
    policy = c.output / "alias-policy.json"
    policy.write_text(
        json.dumps(
            {
                "eligiblePaths": [str(alias)],
                "eligibleTrees": [],
                "keepPaths": [str(keeper)],
                "approval": _reference(approval),
                "terminalProofs": [_reference(terminal)],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = c.output / "child-state/assemble-clean-source.json"
    before = checkpoint.read_bytes()
    request = c.output / "retire.json"
    request.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "verified-hardlink-publication-retirement",
                "publicationOperation": _reference(c.operation),
                "alias": str(alias),
                "keeper": str(keeper),
                "policy": _reference(policy),
                "inventory": _reference(inventory),
                "assembly": {
                    "assemblyManifest": _reference(c.output / "finalized/manifest.json"),
                    "assemblyReceipt": _reference(c.output / "evidence/receipt.json"),
                    "cleanIndex": _reference(c.output / "evidence/single-source-index.json"),
                },
                "childReceipt": _reference(checkpoint),
                "captures": [_reference(terminal)],
                "leases": [str(c.output / ".workflow.lock")],
                "currentReferences": [_reference(c.operation)],
                "output": str(c.output / "alias-retired.json"),
            }
        ),
        encoding="utf-8",
    )
    digest = file_digest(request)
    result = publication.execute(request, sha256=digest)
    assert not alias.exists()
    assert keeper.stat().st_nlink == 1
    assert checkpoint.read_bytes() == before
    assert publication.execute(request, sha256=digest, verify=True) == result
