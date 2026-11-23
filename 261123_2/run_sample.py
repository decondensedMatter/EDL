"""Run sample checks, frequency sweeps and optimized charging for one connected specimen."""
import argparse
import json
from pathlib import Path

from edl_charge import workflow
from edl_charge.session import run, template
from edl_charge.session_backend import preflight


def config_path(sample, use_current_amp):
    path_tag = "with_amp" if use_current_amp else "without_amp"
    return Path(__file__).resolve().parent / "session_configs" / f"{sample}_{path_tag}_updated.json"


def main(sample, use_current_amp, argv, *, default_config=None):
    parser = argparse.ArgumentParser(description=f"{sample}: {__doc__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init-session", "check", "run"):
        cmd = commands.add_parser(name)
        if name != "init-session":
            cmd.add_argument("--config", type=Path, help="Individual bench session JSON")
        if name != "check":
            cmd.add_argument("--out", type=Path, help="Template filename" if name == "init-session"
                             else "Parent directory for a new acquisition")
        cmd.add_argument("--budget-seconds", type=float, help="Active seconds, including calibration and acquisition")
    args = parser.parse_args(argv)
    if args.budget_seconds is not None and not 5 < args.budget_seconds <= 7200:
        parser.error("Choose more than 5 and at most 7200 active seconds (5 seconds reserved for cleanup)")
    default_path = Path(default_config) if default_config is not None else config_path(sample, use_current_amp)
    initializing = args.command == "init-session"
    path = (args.out if initializing else args.config) or default_path
    first_use = not initializing and args.config is None and not path.exists()
    if initializing or first_use:
        data = template(use_current_amp, "updated", selected_samples=[sample],
                        budget_seconds=args.budget_seconds or 1200.,
                        calibration_mode="external_standards" if use_current_amp else "sample_only_nominal")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as output:
                json.dump(data, output, indent=2, default=workflow.serial, allow_nan=False)
                output.write("\n")
        except FileExistsError:
            if initializing:
                parser.error(f"Configuration already exists: {path}. Choose a new --out filename.")
        else:
            print(f"Created individual bench configuration: {path}")
        if initializing:
            return 0
    try:
        config = workflow.read(path)
    except (OSError, ValueError) as exc:
        parser.error(f"Unable to read {path}: {exc}. Create a template with --sample-session init-session --out FILE")
    if (not isinstance(config, dict) or config.get("selected_samples") != [sample]
            or not isinstance(config.get("samples"), dict) or set(config["samples"]) != {sample}
            or config.get("use_current_amp") is not use_current_amp or config.get("readout_mode") != "updated"):
        parser.error(f"Expected an individual updated-readout session configuration for {sample} and this current path")
    if args.budget_seconds is not None:
        config["budget_seconds"] = args.budget_seconds
    issues = preflight(config)
    if issues:
        print(f"Resolve these configuration issues in {path} before acquisition:")
        print("\n".join(issues))
        return 1
    if args.command == "check":
        print(f"Configuration ready for {sample}. Calibration mode: {config.get('calibration_mode', 'standards')}.")
        return 0
    folder = run(config, (args.out or Path("session_data")).resolve())
    journal_path = folder / "session_journal.json"
    journal = workflow.read(journal_path)
    status = journal["status"]
    print(f"Session {status}. Journal: {journal_path}")
    reason = journal.get("error") or journal.get("reason") or journal.get("samples", {}).get(sample, {}).get("reason")
    if reason:
        print(reason)
    return 0 if status == "completed" else 2
