"""Run original coursework scripts on a small synthetic fixture, never course data."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parents[1] / 'analysis'
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory(prefix='library-complexity-demo-', dir=output) as temp:
        work = Path(temp)
        (work / 'data').mkdir()
        for i, name in enumerate(['Sample1_pt1', 'Sample1_pt2', 'Sample1_pt3', 'Sample2']):
            lengths = np.repeat(np.arange(150, 180, 5), 80)
            starts = np.arange(len(lengths)) * 1000
            counts = 1 + rng.poisson(0.5 + i * 0.15 + (lengths - 150) / 60)
            pd.DataFrame({'chr': 'synthetic', 'start': starts,
                          'end': starts + lengths - 1, 'n': counts}).to_csv(
                              work / 'data' / f'{name}.csv', index=False)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1')
        logs = []
        for script in ['q2a.py', 'q2b.py', 'q3a.py', 'q3b.py', 'q3d.py']:
            result = subprocess.run([sys.executable, str(scripts / script)],
                                    cwd=work, env=env, capture_output=True,
                                    text=True, encoding='utf-8', check=True)
            logs.append(f'=== {script} (SYNTHETIC DATA) ===\n{result.stdout}\n{result.stderr}')
        import shutil
        for figure in work.glob('*.pdf'):
            shutil.copy2(figure, output / figure.name)
    (output / 'run.log').write_text('\n'.join(logs), encoding='utf-8')
    (output / 'README.txt').write_text(
        'SYNTHETIC DEMONSTRATION ONLY. These files check the original script execution route.\n'
        'They are not the supplied-data results in the report and figures directories.\n',
        encoding='utf-8')
    print(f'All five scripts completed on synthetic data. Demonstration output: {output}')


if __name__ == '__main__':
    main()
