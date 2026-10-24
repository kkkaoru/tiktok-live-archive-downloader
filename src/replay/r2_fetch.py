"""Fetch a recording's offloaded files back from R2, verified, without overwriting anything.

    python -m replay.r2_fetch RECORDING --role delivery [--path LOCAL_PATH]

The object is read through a temporary authenticated Worker bound to the media bucket (its
upload token is generated per session and never printed); see ``replay.delivery_offload``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from replay import delivery_offload, r2_worker
from replay.core import ReplayError

Session = Callable[[str], AbstractContextManager[delivery_offload.Store]]


def fetch(
    root: Path,
    *,
    role: str,
    path: Path | None = None,
    session: Session = r2_worker.session,
) -> list[Path]:
    with session(delivery_offload.BUCKET) as store:
        return delivery_offload.fetch(root, role=role, store=store, path=path)


def fetch_copy(
    root: Path, *, role: str, into: Path, session: Session = r2_worker.session
) -> list[Path]:
    """Download a role's offloaded files into ``into`` (verified copies; nothing is put back)."""
    entries = [entry for entry in delivery_offload.load(root) if entry.role == role]
    if not entries:
        raise ReplayError(f"No offloaded {role} recorded for {root.name}")
    with session(delivery_offload.BUCKET) as store:
        return [
            delivery_offload.download(entry, store, into / Path(entry.local_path).name)
            for entry in entries
        ]


def restore_if_offloaded(path: Path, *, root: Path, session: Session = r2_worker.session) -> bool:
    """Fetch ``path`` (a file or a hardlink of one) back when it is missing but recorded as
    offloaded, with its links; ``False`` otherwise."""
    if path.exists():
        return False
    entry = delivery_offload.offloaded(root, path)
    if entry is None:
        return False
    fetch(root, role=entry.role, path=Path(entry.local_path), session=session)
    return True


def main(argv: Sequence[str] | None = None, *, session: Session = r2_worker.session) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--role", choices=sorted(delivery_offload.ROLES), required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--edit-root", type=Path, help="the recording's edit root, if not edits/")
    parser.add_argument("--into", type=Path, help="download verified copies here instead")
    args = parser.parse_args(argv)
    root = args.edit_root or delivery_offload.edit_root(args.recording)
    restored = (
        fetch(root, role=args.role, path=args.path, session=session)
        if args.into is None
        else fetch_copy(root, role=args.role, into=args.into, session=session)
    )
    print(json.dumps({"restored": [str(path) for path in restored]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
