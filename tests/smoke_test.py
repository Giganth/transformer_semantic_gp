"""Smoke test: one generation of standard GP and of each transformer operator.

Checks that the package imports, the datasets and the pretrained models load, the variation
operators produce offspring, the evaluation runs, and the monitor CSV is written. It is meant
for CI, so it uses a small problem, a population of 10 and a single generation.

Run from the repository root:  python tests/smoke_test.py
"""
import os
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

RUNS = [
    ("standard_gp", "examples/standard_gp.yaml"),
    ("transformer_gp", "examples/transformer_gp_TSGP.yaml"),
    ("transformer_gp", "examples/transformer_gp_TSGP-2.yaml"),
]

FORBIDDEN_COLUMNS = ("lambda", "delta_lambda", "target_lambda")


def _set(node, key, value):
    """Set ``key`` wherever it appears in a nested config."""
    found = False
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                found = True
            elif isinstance(v, (dict, list)):
                found = _set(v, key, value) or found
    elif isinstance(node, list):
        for item in node:
            found = _set(item, key, value) or found
    return found


def run(name, config_path, out_dir):
    with open(ROOT / config_path) as handle:
        config = yaml.safe_load(handle)

    assert _set(config, "max_generations", 1), f"{config_path}: no max_generations"
    config["popsize"] = 10
    config["console_out"] = False
    config["monitor_folder"] = str(out_dir)
    config.setdefault("problem", {})["dataset"] = "1027_ESL"

    config_file = out_dir / f"{name}_{pathlib.Path(config_path).stem}.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = "-1"   # CI has no GPU
    result = subprocess.run(
        [sys.executable, "examples/symbolic_regression.py", "--config", str(config_file)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"FAILED {config_path}: exit {result.returncode}")

    csv_files = list(out_dir.glob("monitor_run_*.csv"))
    assert csv_files, f"{config_path}: no monitor CSV written"

    header = csv_files[0].read_text().splitlines()[0]
    for column in FORBIDDEN_COLUMNS:
        assert column not in header, f"{config_path}: unexpected column {column!r}"

    lines = csv_files[0].read_text().strip().splitlines()
    assert len(lines) >= 2, f"{config_path}: monitor CSV has no rows"
    print(f"ok  {config_path}  ({len(lines) - 1} generations, {len(header.split(','))} columns)")


def main():
    for _, config_path in RUNS:
        assert (ROOT / config_path).is_file(), f"missing {config_path}"
    with tempfile.TemporaryDirectory() as tmp:
        for name, config_path in RUNS:
            out_dir = pathlib.Path(tmp) / f"{name}_{pathlib.Path(config_path).stem}"
            out_dir.mkdir()
            run(name, config_path, out_dir)
    print("smoke test passed")


if __name__ == "__main__":
    main()
