# MedVol-R1

A source release draft for medical volume referring segmentation using
Qwen3-VL supervised fine-tuning and GRPO. It preserves the local EasyR1
dataset adaptation, task reward implementations, and MedSAM2 NPZ predictor.

## Included workflows

| Task | SFT | GRPO reward | RL setup in this draft |
| --- | --- | --- | --- |
| KiTS23 | Qwen3-VL-4B LoRA, rank 64, 2 epochs, 64 slices, 256² pixels | format 0.30 + spatial 0.35 + temporal 0.35 | full actor update, 4 training GPUs, 3 rollouts |
| CTOrg | Qwen3-VL-4B LoRA configuration draft, rank 64, 2 epochs, 336² pixels | format + spatial + temporal + MedSAM2 consistency, each 0.25 | LoRA rank 128, 2 training GPUs plus 1 reward GPU, 4 rollouts |

KiTS23 parameters are derived from the current local training shell.
CTOrg SFT is a reconstruction from the common config and checkpoint name;
its exact training command still requires author verification. The CTOrg
RL recipe comes from the existing LoRA shell. These are distinct reward
variants, not interchangeable recipes. Full GPU reproduction has not been
run on this publication copy.

## Layout

- `third_party/EasyR1`: training framework, custom CT dataset and rewards.
- `third_party/LLaMA-Factory`: SFT framework and inference scripts.
- `third_party/MedSAM2`: local source for the NPZ predictor; no weights.
- `configs`: explicit task configs with shell overrides folded in.
- `scripts`: portable SFT, export, GRPO and inference entry points.
- `data_preparation`: volume processing, 64-slice selection, SFT and parquet generation.
- `evaluation`: original grounding and CTOrg segmentation evaluation code.
- `docs`: provenance, path overrides and release review notes.

## Environments

Use separate Python 3.10+ environments for SFT/inference and RL. Install a
CUDA-compatible PyTorch build first, matching your machine. The commands
below preserve the source dependency constraints; they are not a tested
lockfile. Flash Attention installation may require the CUDA toolkit.

SFT/inference environment, from the repository root:

```bash
python -m pip install -e third_party/LLaMA-Factory
python -m pip install vllm qwen-vl-utils
```

RL environment:

```bash
python -m pip install -e third_party/EasyR1
python -m pip install -r requirements-rl-extra.txt
SAM2_BUILD_CUDA=0 python -m pip install -e third_party/MedSAM2
```

MedSAM2 requires NumPy >=2.0.1, while LLaMA-Factory requires NumPy <2.
Do not install both frameworks in one environment. The optional SAM2 CUDA
extension is disabled in the example; some mask postprocessing operations
may be limited without it. For compilation, use the bundled CUDA source and
an appropriate toolkit. Run `python -m pip check` in each environment.

Data preparation needs `requirements-data.txt`; KiTS23 raw conversion also
uses SimpleITK. Choose datasets <=4.0.0 for the supplied `Sequence` API.

## Data

Raw scans, labels, patient records, splits and case-derived metadata are
not bundled. Prepare the following local layout:

```text
data/
  ctorg/
    dataset_info.json
    ct_org_frames_refseg_train.json
    ct_org_frames_refseg_test.json
    ct_org_npy/volume-*/image.npy, mask_*.npz
    ct_org_png/volume-*/slice_*.png, text.json
    summary_all_train.csv, summary_all_test.csv
    verl_parquet/ctorg_refseg_verl_train.parquet
  kits23/
    dataset_info.json
    kits23_frames_refseg_train.json
    kits23_frames_refseg_test.json
    split.json
    final_vqa_gen/entity_with_templates.json
    kits23_npy_m3d/case_*/image.npy, mask_*.npz
    kits23_png_m3d/case_*/png_64slices/slice_*.png, text.json
    verl_parquet/kits23_refseg_verl_{train,test}.parquet
```

Copy the corresponding `data_preparation/<task>/dataset_info.json` into
`data/<task>/`. The scripts preserve the original sampling, template
whitelist and coordinate conventions. Run them from the repository root;
configure input/output paths through the environment variables listed in
`docs/path_overrides.json`. Output defaults are aligned to `data/<task>/`; environment overrides are
listed alongside the input defaults.
The included term dictionary and template JSON contain generic prompts,
not case annotations. Split files and metadata must be supplied separately.

CTOrg stages: `convert_volumes.py` → `select_slices.py` →
`generate_sft.py` / `generate_rl.py`. For KiTS23, use the entity generation
scripts first, then `step0_cvt_to_kits_m3d.py` → `step1_get_64slices.py` →
`step2_gen_sft_vqa_via_tmplate.py` / `gen_rl_parquet.py`. The entity scripts
expect the original raw data layout; inspect their documented path overrides.
CTOrg candidate CSV creation and dataset split selection are external inputs.

RL parquet includes `problem`, `images`, and `answer` fields. `answer` holds
selected slice IDs, mask/image relative paths, target label or template index,
and bounding boxes. Image paths are generated as absolute paths on the
machine performing preprocessing; regenerate parquet after moving data.
Mask sparse-array shapes/flattening conventions must match each reward.

## Train and infer

In the SFT environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_sft.sh kits23
python scripts/export_sft.py --adapter outputs/sft/kits23/checkpoint-724 \
  --output outputs/merged/kits23_sft
```

Replace `checkpoint-724` with the checkpoint actually produced by your run.

In the RL environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_grpo.sh kits23
bash scripts/export_rl.sh full checkpoints/rl/kits23/global_step_50/actor
```

The full exporter writes the Hugging Face model under the actor's
`huggingface` subdirectory. Use the actual produced checkpoint step.

CTOrg also needs `checkpoints/MedSAM2_2411.pt` (download separately), or set
`MEDSAM2_CHECKPOINT`. Reserve three visible GPUs for the default 2-GPU
training pool and the reward actor:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_grpo.sh ctorg
bash scripts/export_rl.sh lora checkpoints/rl/ctorg/global_step_20/actor \
  outputs/merged/ctorg_sft outputs/merged/ctorg_rl
```

SFT uses answer-only targets. RL requires
`<think>...</think><answer>...</answer>`; the answer is a JSON list such as
`[{"slice": 12, "bbox_2d_list": [[100, 200, 300, 400]]}]`. Coordinates are
normalized to [0,1000], and slice tags use source-volume indices.

Back in the SFT/inference environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh kits23 \
  checkpoints/rl/kits23/global_step_50/actor/huggingface
```

Entry scripts accept extra framework arguments. For example,
`bash scripts/train_sft.sh kits23 learning_rate=1e-5`.
Validation is disabled in the imported GRPO recipes. CTOrg retains the
training parquet in `val_files` because no CTOrg validation parquet was
found; it must not be reported as held-out evaluation.

## Evaluate and publish

Original grounding evaluators are under `evaluation/grounding`; CTOrg
MedSAM2 propagation and mask metrics are in `evaluation/ctorg_segmentation.py`.
Configure path overrides and GPU IDs before running. CTOrg defaults to one
worker on visible GPU 0; use `MEDVOL_EVAL_GPU_IDS`, `MEDVOL_EVAL_NUM_WORKERS`
and `MEDVOL_EVAL_WORKERS_PER_GPU` to adjust it. The grounding scripts
expect their original per-case `text.json` metadata; applicability to each
final dataset still needs verification. Do not treat them as a verified
KiTS23 segmentation benchmark.

Run `python scripts/check_release.py` for static validation and
`python scripts/smoke_kits23_reward.py` for the CPU synthetic reward check.
See `docs/RELEASE_REVIEW.md` for remaining scientific/publication decisions.
No GitHub upload is performed by these commands.

Upstream license files are retained. The repository homepage provides the paper and citation. A root LICENSE
file for original project additions and model download links still need to
be provided. See `THIRD_PARTY_NOTICES.md`.
