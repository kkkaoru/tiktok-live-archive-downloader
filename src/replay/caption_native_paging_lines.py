"""Run native paging, or its verifier, with a lower page line limit for a renderer that needs it.

Paging uses the style's line limit (``caption_style_defaults.MAX_CAPTION_LINES``) and records
it in the reflow (``maximumLines``); its verifier accepts only that recorded limit. The
top-anchored Motion route renders two text layers, so its pages must hold at most two lines;
otherwise placement or the Motion export has to drop the longer pages. This entry point runs one
of those modules with every holder of the limit lowered, so paging and its verifier agree on the
lower limit; the modules themselves, pinned by earlier evidence, are unchanged. Paging also keeps
a page's line breaks once shown (``replay.caption_stable_lines``).

    python -m replay.caption_native_paging_lines --maximum-lines 2 [--module M] -- ARGUMENTS
"""

import runpy
import sys
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager

from replay import caption_style_defaults
from replay.caption_stable_lines import kept
from replay.core import ReplayError

LIMIT = "MAX_CAPTION_LINES"
MODULES = ("replay.caption_native_paging", "replay.caption_reflow_verify")
USAGE = "Usage: caption_native_paging_lines --maximum-lines N [--module M] -- ARGUMENTS"


def limit_of(arguments: Sequence[str]) -> tuple[int, str, list[str]]:
    """The requested line limit, the module to run and its arguments after ``--``."""
    if "--" not in arguments:
        raise ReplayError(USAGE)
    split = list(arguments).index("--")
    options, rest = list(arguments[:split]), list(arguments[split + 1 :])
    module = MODULES[0]
    if len(options) == 4 and options[2] == "--module":
        module, options = options[3], options[:2]
    if len(options) != 2 or options[0] != "--maximum-lines" or not options[1].isdigit():
        raise ReplayError(USAGE)
    if module not in MODULES:
        raise ReplayError(f"Unsupported module: {module}")
    limit = int(options[1])
    if not 1 <= limit <= caption_style_defaults.MAX_CAPTION_LINES:
        raise ReplayError("The line limit must be within the style's limit")
    return limit, module, rest


def holders(value: int, names: Iterable[str] | None = None) -> list[str]:
    """Loaded replay modules (optionally among ``names``) whose line limit equals ``value``."""
    chosen = set(sys.modules) if names is None else set(names)
    return [
        name
        for name, module in list(sys.modules.items())
        if name in chosen and name.startswith("replay.") and module.__dict__.get(LIMIT) == value
    ]


@contextmanager
def lowered(limit: int) -> Generator[None]:
    """Every holder sees ``limit`` inside the block; the style's limit is restored after it.

    Modules imported inside the block read the lowered style value when they import it, so
    they are restored as well.
    """
    style = caption_style_defaults.MAX_CAPTION_LINES
    before = set(sys.modules)
    changed = holders(style)
    for name in changed:
        sys.modules[name].__dict__[LIMIT] = limit
    try:
        yield
    finally:
        imported = holders(limit, set(sys.modules) - before)
        for name in [*changed, *imported]:
            sys.modules[name].__dict__[LIMIT] = style


def main(argv: Sequence[str] | None = None) -> None:
    limit, module, arguments = limit_of(sys.argv[1:] if argv is None else argv)
    saved = sys.argv
    sys.argv = [module, *arguments]
    try:
        with lowered(limit), kept():
            runpy.run_module(module, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
