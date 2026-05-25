<div align="center">

# MedVol-R1: Reward-Driven Evidence Grounding for Volumetric Reasoning Segmentation

<p>
  <a href="#"><img src="https://img.shields.io/badge/MICCAI-2026-blue.svg" alt="MICCAI 2026"></a>
  <a href="#"><img src="https://img.shields.io/badge/Provisional%20Accept-Top%209%25-brightgreen.svg" alt="Top 9%"></a>
  <a href="#"><img src="https://img.shields.io/badge/Task-Volumetric%20Reasoning%20Segmentation-orange.svg" alt="Task"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

**Zichun Wang**<sup>1†</sup>, **Hairong Shi**<sup>2†</sup>, Bingzheng Wei<sup>3</sup>, Yan Xu<sup>1✉</sup>, Zihua Wang<sup>4✉</sup>

<sup>1</sup>Beihang University &nbsp; <sup>2</sup>Keio University &nbsp; <sup>3</sup>ByteDance Inc. &nbsp; <sup>4</sup>Tsinghua University

<sup>†</sup>Equal contribution &nbsp;&nbsp; <sup>✉</sup>Corresponding authors

</div>

---

## News

- 🎉 **[2026-05]** MedVol-R1 is **provisionally accepted at MICCAI 2026 (Top 9%)**.
- 🎉 **[2026-05]** Project page and code repository are now live.

## Overview

**Volumetric Reasoning Segmentation (VRS)** aims to segment a target region in a 3D medical scan from a *free-form clinical query*, where the referent is often implicit and requires both medical knowledge (e.g., *"the organ that secretes bile"*) and volume-grounded reasoning (e.g., *"the kidney that contains the tumor"*).

Existing LVLM-based methods rely on specialized segmentation tokens (e.g., `<SEG>`) that collapse the decision process into opaque latent representations, limiting interpretability and generalization to diverse narrative expressions.

We present **MedVol-R1**, the *first reinforcement-learning–based framework* for VRS that explicitly **decouples evidence grounding from volumetric delineation**:

- The **LVLM** grounds clinical reasoning to a *verifiable 2D evidence anchor* — a key axial slice plus 2D bounding boxes.
- A **frozen MedSAM2** propagates this anchor into a coherent 3D mask.
- Training combines **cold-start SFT** with **GRPO**, guided by a multi-component reward — without any chain-of-thought annotation.

<p align="center">
  <img src="asset/Fig1.png" alt="MedVol-R1 Pipeline" width="95%">
</p>
<p align="center"><i>Figure 1. Overall pipeline of MedVol-R1. The reasoning policy produces a key-slice and bounding boxes, which are scored by a multi-dimensional rule-based reward and propagated to a 3D mask by a frozen MedSAM2.</i></p>


## Results

### Quantitative Comparison

MedVol-R1 (SFT+RL) achieves the **best DSC and IoU on all three benchmarks**, with the largest improvements on reasoning-heavy queries.

<p align="center">
  <img src="asset/Tab1.png" alt="Quantitative comparison" width="80%">
</p>
<p align="center"><i>Table 1. Quantitative comparison with state-of-the-art methods on AbdomenCT-1K, CT-ORG, and KiTS23.</i></p>

### Qualitative Comparison

<p align="center">
  <img src="asset/Fig2.png" alt="Qualitative comparisons" width="95%">
</p>
<p align="center"><i>Figure 2. Qualitative comparisons on five representative VRS samples. MedVol-R1 yields cleaner boundaries, less leakage into adjacent structures, and more complete target coverage than M3D.</i></p>

## Citation

The official MICCAI 2026 proceedings citation is **to be released**. A placeholder BibTeX entry is provided below and will be updated once the camera-ready version is published:

```bibtex
@inproceedings{wang2026medvolr1,
  title     = {MedVol-R1: Reward-Driven Evidence Grounding for Volumetric Reasoning Segmentation},
  author    = {Wang, Zichun and Shi, Hairong and Wei, Bingzheng and Xu, Yan and Wang, Zihua},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {To appear}
}
```

## Acknowledgments

This work is supported by the National Natural Science Foundation of China under Grants 62371016 and U23B2063, the Beijing Natural Science Foundation Haidian District Joint Fund under Grant L222032, the Fundamental Research Funds for the Central University of China from the State Key Laboratory of Software Development Environment in Beihang University, the 111 Project under Grant B13003, SinoUnion Healthcare Inc. under the eHealth program, and HPC resources at Beihang University.

We thank the authors of **[M3D](https://github.com/BAAI-DCAI/M3D.git)**, **[MedSAM2](https://github.com/bowang-lab/MedSAM2)**, **[SAT](https://github.com/zhaoziheng/SAT)** and **[BiomedParseV2](https://github.com/microsoft/BiomedParse.git)** for releasing their work that this project builds upon.

## Contact

For questions, please reach out to the corresponding authors:

- Yan Xu — `xuyan04@gmail.com`
- Zihua Wang — `wangzihua07@126.com`
