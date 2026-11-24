"""Run regression tests and real-checkpoint application checks."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-only', action='store_true', help='Skip checks requiring saved checkpoints')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = ['check_preparation', 'check_telugu_comparison', 'check_telugu_cnn', 'check_wordchar']
    if not args.unit_only:
        checks += ['check_telugu_ui', 'check_final_submission']
    subprocess.run([sys.executable, '-m', 'compileall', '-q', 'src', 'tests'], cwd=root, check=True)
    for check in checks:
        subprocess.run([sys.executable, '-X', 'utf8', str(root / 'tests' / f'{check}.py')], cwd=root, check=True)
    print(f'PASS: syntax and {len(checks)} check suites')


if __name__ == '__main__':
    main()
