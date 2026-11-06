"""Command line entry points; input roots and all network paths are read-only."""

import argparse, os
from pathlib import Path


def main():
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "analyse", "ei", "wells"]:
        p = sub.add_parser(name)
        p.add_argument("--native-root", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        if name == "analyse":
            p.add_argument("--trees", type=int, default=500)
            p.add_argument("--skip-embeddings", action="store_true")
            p.add_argument("--skip-wells", action="store_true")
        elif name == "validate":
            p.add_argument("--limit", type=int)
        elif name == "ei":
            p.add_argument("--labels", required=True, type=Path)
    args = parser.parse_args()
    from .safety import install_readonly_guard, is_network_path

    if is_network_path(args.output):
        parser.error("this command writes local outputs only")
    install_readonly_guard(
        [args.native_root] + ([args.labels] if args.command == "ei" else [])
    )
    if args.command == "validate":
        from .validation import validate_native

        rows = validate_native(args.native_root, args.output, limit=args.limit)
        if not all(r["passed"] for r in rows):
            raise SystemExit(1)
    elif args.command == "wells":
        from .report import publish_wells

        publish_wells(args.native_root, args.output)
    elif args.command == "ei":
        from .ei_workflow import build_ei_reports

        build_ei_reports(args.native_root, args.labels, args.output)
    else:
        from .report import build_reports

        build_reports(
            args.native_root,
            args.output,
            trees=args.trees,
            do_embeddings=not args.skip_embeddings,
            do_wells=not args.skip_wells,
        )


if __name__ == "__main__":
    main()
