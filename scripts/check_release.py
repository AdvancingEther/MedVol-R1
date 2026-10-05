"""Check selected source syntax, configs, entry scripts and excluded artifacts."""
import ast
import json
import re
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    python_files = [p for p in files if p.suffix == '.py']
    for p in python_files:
        try:
            ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as exc:
            errors.append(str(exc))
    for p in files:
        if p.suffix in {'.yaml', '.yml'}:
            try:
                yaml.safe_load(p.read_text())
            except yaml.YAMLError as exc:
                errors.append(f'{p}: {exc}')
        if p.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(p)], capture_output=True, text=True)
            if result.returncode:
                errors.append(result.stderr)
        if p.suffix == '.json':
            try:
                json.loads(p.read_text())
            except ValueError as exc:
                errors.append(f'{p}: {exc}')
        rel = p.relative_to(ROOT)
        if p.name.startswith('.env') and p.name != '.env.example':
            errors.append(f'Excluded environment file: {rel}')
        if p.suffix in {'.pt', '.pth', '.bin', '.npy', '.npz', '.parquet', '.safetensors', '.jsonl', '.png', '.jpg'} and str(rel) not in {'asset/Fig1.png', 'asset/Fig2.png', 'asset/Tab1.png', 'asset/Tab2.png'}:
            errors.append(f'Excluded data/weight/output: {rel}')
        if p.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f'Oversized: {rel}')
        if p.suffix in {'.py', '.yaml', '.sh', '.json', '.md'}:
            text = p.read_text()
            if re.search(r'/home/[a-zA-Z0-9_-]+/', text):
                errors.append(f'Machine-specific path: {rel}')
            if re.search(r'(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_]{24,}', text):
                errors.append(f'Possible credential: {rel}')
    for task in ['ctorg', 'kits23']:
        c = yaml.safe_load((ROOT / f'configs/grpo_{task}.yaml').read_text())
        reward = c['worker']['reward']['reward_function'].split(':')[0]
        if not (ROOT / reward).is_file():
            errors.append(f'Missing reward: {reward}')
        if c['trainer']['val_freq'] > 0 or c['trainer']['val_before_train']:
            errors.append('Imported validation must stay disabled pending split review')
    report = dict(files=len(files), python_files=len(python_files),
                  bytes=sum(p.stat().st_size for p in files), errors=errors)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
