# Release review

## Completed

- Selected code copied into a separate release directory; experimental sources unchanged.
- Both frameworks and local MedSAM2 predictor preserved without nested Git history.
- Current KiTS23 shell identified despite its CTOrg filename.
- Explicit task configs and portable entry points added.
- Machine-specific Python path constants converted to documented environment overrides.
- Raw data, predictions, visualization outputs, credentials and weights excluded.
- Third-party licenses, source revisions and source hashes retained.

## Author decisions / verification before public reproduction claims

1. Confirm which CTOrg and KiTS23 checkpoints/configurations correspond to the paper.
   CTOrg SFT config is reconstructed, not an independently recovered exact recipe.
2. Confirm whether CTOrg full-update or LoRA GRPO is the primary reported experiment.
   This draft preserves the recoverable CTOrg LoRA and KiTS23 full-update recipes.
3. Supply permitted dataset download instructions, split generation, CTOrg candidate
   CSVs, and any case-derived metadata needed by preprocessing.
4. Confirm final evaluation scripts. The supplied grounding evaluators originated
   from earlier multi-entity datasets. A verified KiTS23 final segmentation
   evaluator was not located within the inspected primary directories.
5. Reproduce GPU training/inference, including dataset shapes and reward values.
   Static and synthetic checks do not establish numerical reproduction.
6. Provide a root LICENSE consistent with the existing MIT badge; add checkpoint links.
   The repository homepage already provides paper title, authors and citation.
7. Decide whether MedGemma baselines and BiomedCLIP retrieval belong in this release.

## Changes from experiment copies

KiTS23 reward GPU reservation changed from 1 to 0 because its implementation
uses NumPy/SciPy and does not call SAM2. Training GPU count/rollouts retained.
External WandB endpoint removed; local console logging is the RL default.
SFT overwrite_output_dir and KiTS23 raw converter OVERWRITE disabled to
avoid replacing existing runs/data. Preprocessing output paths aligned.
CTOrg evaluation defaults to one worker on visible GPU 0, with environment
overrides, rather than the original machine GPU IDs 4 and 5.
KiTS23 val_files now points to its generated test parquet, but validation
remains disabled. Do not tune on the test split; provide a separate validation
split before enabling validation. CTOrg val_files stays train, also disabled.
No reward weights, parsing, mask decoding or matching algorithms were changed.

## GitHub steps

The code is integrated into AdvancingEther/MedVol-R1, preserving its
existing paper README and published figures. Inspect the staged list and
push from the separate cloned repository only. Never run git add . in the
parent experiment workspace. Do not import the original nested .git histories: .env.local was
tracked in the LLaMA-Factory source, so fresh source selection avoids carrying
that file or its history into this release.
