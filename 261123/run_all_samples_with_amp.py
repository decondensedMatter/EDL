"""Six samples, current amplifier and updated readout, one 120-minute budget."""
from run_all_samples import main

if __name__=='__main__':
    raise SystemExit(main(use_current_amp=True,readout_mode='updated'))
