"""Prepare and qualify EDL waveform files offline. This entry point never connects to instruments."""
import argparse
from pathlib import Path

from edl_charge import workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("plan", "Write voltage and duration combinations and calibration recipes."),
        ("init-config", "Write an editable specimen-specific calibration configuration."),
        ("fit", "Fit and qualify the sample model against independent calibration traces."),
        ("optimize", "Export candidate charging controls and compiled-model verification."),
        ("demo", "Write a provisional historical 0.1 M example for inspection only."),
    ):
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--out", required=True, type=Path)
        if name in ("plan", "init-config"):
            sub.add_argument("--sample", required=name == "init-config")
            sub.add_argument("--current-path", required=True, choices=("with-amp", "without-amp"))
        if name in ("fit", "optimize"):
            sub.add_argument("--config", required=True, type=Path)
        if name == "fit":
            sub.add_argument("--manifest", required=True, type=Path)
        if name == "optimize":
            sub.add_argument("--limit", type=int, help="Maximum number of voltage-duration pairs.")
    args = parser.parse_args(argv)
    if args.command == "plan":
        workflow.save(args.out, workflow.plan(args.sample, args.current_path == "with-amp"))
    elif args.command == "init-config":
        from edl_charge.model_defaults import model_fit_defaults
        workflow.save(args.out, model_fit_defaults(workflow.make_config(args.sample, args.current_path == "with-amp")))
    elif args.command == "fit":
        workflow.save(args.out, workflow.fit(workflow.read(args.config), args.manifest))
    elif args.command == "optimize":
        config = workflow.read(args.config)
        workflow.optimize(config, args.out, args.limit)
        workflow.save(args.out / "fitted_config.json", config)
    elif args.command == "demo":
        workflow.demo(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
