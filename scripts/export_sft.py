"""Export an SFT LoRA adapter using the installed LLaMA-Factory environment."""
import argparse
import subprocess
import tempfile
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be absent or empty.")
    config = dict(model_name_or_path=args.base_model,
                  adapter_name_or_path=str(Path(args.adapter).resolve()),
                  export_dir=str(output), template="qwen3_vl", trust_remote_code=True,
                  export_size=5, export_device="cpu", export_legacy_format=False)
    with tempfile.TemporaryDirectory(prefix="medvol-export-") as temp:
        config_path = Path(temp) / "export.yaml"
        config_path.write_text(yaml.safe_dump(config))
        subprocess.run(["llamafactory-cli", "export", str(config_path)], check=True)


if __name__ == "__main__":
    main()
