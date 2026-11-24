"""Export the committed Git tree, excluding local data and ignored artifacts."""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from text_utils import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = project_root()
    output = args.output or root / f'next-word-prediction-{datetime.now():%Y%m%d-%H%M%S}.zip'
    output = output.resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}')
    subprocess.run(['git', 'archive', '--format=zip', f'--output={output}', 'HEAD'], cwd=root, check=True)
    print(f'Exported committed HEAD to {output}; uncommitted changes are not included.')


if __name__ == '__main__':
    main()
