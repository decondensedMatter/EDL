"""Dispatch a fixed specimen and current path to single-shot chirp or EDL acquisition."""
import argparse
import sys

from edl_charge.planning import get_profile


def _chirp(profile, use_current_amp, args):
    from keysight_33511b_chirp_scan import ScanSettings, print_plan, run_scan, scan_plans
    parser=argparse.ArgumentParser(prog="updated-readout chirp",description="Acquire a finite two-round-trip sweep.")
    parser.add_argument("--preview",action="store_true",help="Print planned sweeps without instrument access.")
    parser.add_argument("--mode",choices=("native","arb"),default="native")
    parser.add_argument("--leg-duration-s",type=float,default=1.)
    parser.add_argument("--out",help="Acquisition output root; defaults to the existing chirp readout location.")
    opts=parser.parse_args(args)
    sample=profile["sample_name"]
    chirp_name=("Ta_LLZO" if sample=="LLZO" else
                sample.split("_")[1]+" NaClO4" if sample.startswith("NaClO4_") else sample)
    settings=ScanSettings(sample_name=chirp_name,device_name=sample,sweep_mode=opts.mode,
                          native_leg_duration_s=opts.leg_duration_s,arb_leg_duration_s=opts.leg_duration_s,
                          use_current_amp=use_current_amp,electrode_area_m2=profile["model_parameters"]["area_m2"],
                          current_preamplifier_gain_a_per_v=.001 if use_current_amp else .02)
    if opts.out:
        settings.base_save_dir=opts.out
    if opts.preview:
        print_plan(settings,scan_plans(settings))
    else:
        run_scan(settings)
    return 0


CAMPAIGN_FLAGS = ("--campaign-six","--session")


def campaign_from_entry(sample, use_current_amp, readout_mode, args):
    """Bind the original entry's electrical path and readout to the six-specimen run."""
    from run_all_samples import main as campaign_main
    if readout_mode not in ("updated","legacy") or type(use_current_amp) is not bool:
        raise ValueError("Specify a supported readout mode and current path")
    get_profile(sample)
    return campaign_main(list(args), use_current_amp=use_current_amp, readout_mode=readout_mode)


def main(sample, use_current_amp, argv=None):
    """Recognized commands are chirp, plan, init-config, fit, optimize, run, analyze and acquire-calibration."""
    if type(use_current_amp) is not bool:
        raise ValueError("Current path must be explicitly True or False")
    canonical=get_profile("LLZO" if sample=="Ta_LLZO" else sample)
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in CAMPAIGN_FLAGS:
        return campaign_from_entry(canonical["sample_name"],use_current_amp,"updated",args[1:])
    if not args or args[0] == "--sample-session" or (args[0].startswith("-") and args[0] not in ("-h","--help")):
        from run_sample import main as sample_main
        if args and args[0] == "--sample-session":
            args = args[1:]
        if not args or (args[0].startswith("-") and args[0] not in ("-h","--help")):
            args = ["run", *args]
        return sample_main(canonical["sample_name"],use_current_amp,args)
    commands=("chirp","plan","init-config","fit","optimize","run","run-campaign","analyze","acquire-calibration")
    if not args or args[0] in ("-h","--help"):
        help_parser=argparse.ArgumentParser(description=f'{canonical["sample_name"]} updated readout; '
                                                        f'current amplifier: {use_current_amp}')
        help_parser.epilog=("Default: full individual session. Use --sample-session {init-session,check,run}, "
                            "or --config FILE --out DIR to run. Use --campaign-six for all six specimens.")
        help_parser.add_argument("command",choices=commands)
        help_parser.print_help()
        return 0
    if args[0]=="chirp":
        return _chirp(canonical,use_current_amp,args[1:])
    from edl_charge.cli import main as charge_main
    fixed=canonical["sample_name"]
    if args[0]=="init-config" and not any(k=="--sample" or k.startswith("--sample=") for k in args):
        args.extend(["--sample",fixed])
    return charge_main(use_current_amp,args,fixed_sample=fixed)

