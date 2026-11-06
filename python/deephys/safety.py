"""CLI safety: protected input roots and network locations are read-only."""

from pathlib import Path
import os, sys


def is_network_path(path):
    text = os.fsdecode(os.fspath(path))
    if text.startswith(("\\\\", "//")):
        return True
    if os.name == "nt":
        import ctypes

        drive = Path(text).resolve().anchor
        return bool(drive) and ctypes.windll.kernel32.GetDriveTypeW(drive) == 4
    return False


def install_readonly_guard(input_roots):
    roots = [Path(p).resolve() for p in input_roots]

    def protected(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        if is_network_path(path):
            return True
        p = Path(os.fsdecode(path)).resolve()
        return any(p == root or root in p.parents for root in roots)

    def guard(event, args):
        paths = []
        if event == "open":
            path, mode, flags = args
            writing = (
                isinstance(mode, str) and any(c in mode for c in "wax+")
            ) or bool(
                flags
                & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writing:
                paths = [path]
        elif event in (
            "os.remove",
            "os.rmdir",
            "os.mkdir",
            "os.chmod",
            "os.truncate",
            "os.utime",
        ):
            paths = list(args[:1])
        elif event in ("os.rename", "os.link", "os.symlink"):
            paths = list(args[:2])
        if any(protected(p) for p in paths):
            raise PermissionError("DeePhys: protected input/network mutation denied")

    sys.addaudithook(guard)
    return guard
