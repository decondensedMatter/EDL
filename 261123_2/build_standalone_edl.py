"""Build six self-contained direct-readout launchers from the audited public tree."""
import base64
import hashlib
import io
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'standalone_scripts'
SAMPLES = ['NaClO4_0.01M', 'NaClO4_0.1M', 'NaClO4_1M', 'NaClO4_9M', 'LLZO', 'LaF3']
RUNTIME_NAMES = ['run_sample.py', 'keysight_33511b_chirp_plan.py',
                 'keysight_33511b_chirp_generator.py', 'keysight_33511b_chirp_scan.py']


def build():
    paths = [ROOT / name for name in RUNTIME_NAMES]
    paths += [p for p in (ROOT / 'edl_charge').glob('*.py') if not p.name.startswith('test_') and p.name != 'bench_setup.py']
    paths += list((ROOT / 'edl_charge').glob('*.json'))
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            entry = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 11, 23, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    payload = content.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = '\n'.join('    ' + repr(line) for line in textwrap.wrap(base64.b85encode(payload).decode(), 100))
    OUT.mkdir(exist_ok=True)
    for sample in SAMPLES:
        source = '''"""SAMPLE: unattended, direct 50-ohm sensing, updated readout.

Run this Python file with the specimen connected. No keyboard input or specimen
changes are requested. Hardware conversion/response remain explicitly nominal;
sample-only checks do not independently calibrate the measurement chain.
Default active limit: 20 minutes, including checks, computation and acquisitions.
This file embeds its shared source; no other project files or download are needed.
Python 3.10+, numpy, scipy, matplotlib, PyVISA and a VISA driver are required.
"""
import base64
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import zipfile

_SAMPLE = SAMPLE_REPR
_PAYLOAD_SHA256 = DIGEST_REPR
_PAYLOAD = (
ENCODED
)


def main():
    if sys.version_info < (3, 10):
        print('Python 3.10 or later is required.')
        return 2
    missing = [name for name in ('numpy', 'scipy', 'matplotlib', 'pyvisa')
               if importlib.util.find_spec(name) is None]
    # Offline configuration checks do not need VISA.
    offline = any(arg in ('--help', '-h', 'check', 'init-session') for arg in sys.argv[1:])
    if missing and not (offline and missing == ['pyvisa']):
        print('Missing Python packages: ' + ', '.join(missing))
        print('Install once in this Python environment:')
        print('  python -m pip install numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.8 pyvisa')
        print('Real acquisition also requires the instrument VISA driver.')
        return 2
    payload = base64.b85decode(_PAYLOAD)
    if hashlib.sha256(payload).hexdigest() != _PAYLOAD_SHA256:
        raise RuntimeError('Embedded runtime checksum failed')
    home = Path(__file__).resolve().parent
    runtime = home / '.edl_runtime' / _PAYLOAD_SHA256
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            target = (runtime / entry.filename).resolve()
            if not target.is_relative_to(runtime.resolve()) or entry.is_dir():
                raise RuntimeError('Invalid embedded runtime path')
            expected = archive.read(entry)
            if not target.is_file() or target.read_bytes() != expected:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(expected)
                try:
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
    sys.path.insert(0, str(runtime))
    from run_sample import main as run_sample
    args = sys.argv[1:]
    if not args or (args[0].startswith('-') and args[0] not in ('-h', '--help')):
        args = ['run'] + args
    if args[0] == 'run' and not any(arg == '--out' or arg.startswith('--out=') for arg in args):
        args += ['--out', str(home / 'data')]
    config = home / 'edl_settings' / (_SAMPLE + '_without_amp_updated.json')
    return run_sample(_SAMPLE, False, args, default_config=config)


if __name__ == '__main__':
    raise SystemExit(main())
'''
        source = source.replace('SAMPLE_REPR', repr(sample)).replace('DIGEST_REPR', repr(digest))
        source = source.replace('ENCODED', encoded).replace('SAMPLE:', sample + ':', 1)
        filename = sample.replace('.', 'p') + '_without_current_amp_updated_readout.py'
        (OUT / filename).write_text(source, encoding='utf-8', newline='\n')
    print(f'Built {len(SAMPLES)} launchers; {len(paths)} runtime files; embedded ZIP {len(payload)} bytes; sha256 {digest}')


if __name__ == '__main__':
    build()
