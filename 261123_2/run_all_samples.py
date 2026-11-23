"""Run one six-specimen campaign under a shared active-time budget."""
import argparse
from pathlib import Path

from edl_charge import workflow as w
from edl_charge.session import readout_issues,run,template
from edl_charge.session_backend import model_input_issues,preflight


def config_filename(use_current_amp,readout_mode):
    path='with_amp' if use_current_amp else 'without_amp'
    return f'all_samples_120min_{path}_{readout_mode}.json'


def main(argv=None,*,use_current_amp=None,readout_mode=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('init-session','check','run'):
        cmd=sub.add_parser(name)
        cmd.add_argument('--current-path',choices=('direct','amplifier'))
        cmd.add_argument('--readout',choices=('updated','legacy'))
        if name!='init-session':
            cmd.add_argument('--config')
        if name in ('init-session','run'):
            cmd.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    requested_amp=None if args.current_path is None else args.current_path=='amplifier'
    if use_current_amp is not None and requested_amp is not None and requested_amp is not use_current_amp:
        parser.error('The selected sample entry binds a different current path')
    if readout_mode is not None and args.readout is not None and args.readout!=readout_mode:
        parser.error('The selected sample entry binds a different readout mode')
    amp=use_current_amp if use_current_amp is not None else bool(requested_amp)
    readout=readout_mode or args.readout or 'updated'
    if args.command=='init-session':
        print(w.save(args.out,template(amp,readout)))
        return 0
    config_path=Path(args.config) if args.config else Path(__file__).parent/config_filename(amp,readout)
    config=w.read(config_path)
    if config.get('use_current_amp') is not amp or config.get('readout_mode','updated')!=readout:
        parser.error('Session configuration does not match the selected current path/readout')
    if args.command=='check':
        issues=readout_issues(config)+preflight(config)
        print('\n'.join(issues) if issues else 'Bench inputs present. Calibration will be measured in this session.')
        for name,cfg in config['samples'].items():
            pending=model_input_issues(cfg)
            if pending:
                print(f'{name}: optimization pending model inputs; calibrated raw traces and sweeps remain planned.')
                for reason in pending:
                    print('  '+reason)
        return 1 if issues else 0
    folder=run(config,args.out)
    journal=w.read(folder/'session_journal.json')
    print('Session status:',journal['status'])
    print('Session journal:',folder/'session_journal.json')
    print('Active seconds:',round(journal['active_elapsed_s'],3))
    return 0 if journal['status']=='completed' else 2


if __name__=='__main__':
    raise SystemExit(main())
