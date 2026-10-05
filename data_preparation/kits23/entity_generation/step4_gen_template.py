#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert entity_dict.json to entity_with_templates.json

Revised rules (deduplicated by template inclusion / priority):

Kidney templates: keep only
- kidney_only_with_tumor
- kidney_only_with_cyst
- kidney_only_with_both_tumor_and_cyst
- kidney_only_healthy

Tumor/Cyst templates:
Priority:
1) if is_only_global == True:
   -> ONLY keep "the only tumor/cyst"

2) elif is_only_in_kidney == True:
   -> ONLY keep "the tumor/cyst in kidney_x"

3) else:
   -> consider ONLY kidney-level ranking templates:
      - largest in kidney
      - smallest in kidney
      - highest in kidney
      - lowest in kidney

No global ranking templates in this revised version.
"""

import json
import os
from pathlib import Path


# =========================================================
# Global config
# =========================================================
entity_json = os.environ.get('MEDVOL_STEP4_GEN_TEMPLATE_ENTITY_JSON', 'data/kits23/final_vqa_gen/entity_dict.json')
output_json = os.environ.get('MEDVOL_STEP4_GEN_TEMPLATE_OUTPUT_JSON', 'data/kits23/final_vqa_gen/entity_with_templates.json')


# =========================================================
# Helpers
# =========================================================
def safe_mkdir_for_file(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def add_template(template_list, template_id: str, canonical_query: str, question_type: str):
    template_list.append({
        "template_id": template_id,
        "canonical_query": canonical_query,
        "question_type": question_type,
    })


# =========================================================
# Template matching
# =========================================================
def build_templates_for_kidney(entity: dict):
    """
    Keep only the core kidney state templates.
    """
    templates = []

    if entity.get("is_only_kidney_with_tumor", False):
        add_template(
            templates,
            "kidney_only_with_tumor",
            "the kidney with tumor",
            "organ_reasoning"
        )

    if entity.get("is_only_kidney_with_cyst", False):
        add_template(
            templates,
            "kidney_only_with_cyst",
            "the kidney with cyst",
            "organ_reasoning"
        )

    if entity.get("is_only_kidney_with_both_tumor_and_cyst", False):
        add_template(
            templates,
            "kidney_only_with_both_tumor_and_cyst",
            "the kidney with both tumor and cyst",
            "organ_reasoning"
        )

    if entity.get("is_only_healthy_kidney", False):
        add_template(
            templates,
            "kidney_only_healthy",
            "the healthy kidney",
            "organ_reasoning"
        )

    return templates


def build_templates_for_lesion(entity: dict):
    """
    Revised priority logic for tumor/cyst:

    Priority 1:
      if is_only_global:
        -> only "the only tumor/cyst"

    Priority 2:
      elif is_only_in_kidney:
        -> only "the tumor/cyst in kidney_x"

    Priority 3:
      else:
        -> only kidney-level ranking templates:
            - largest in kidney
            - highest in kidney
            - lowest in kidney

    No smallest templates.
    No global ranking templates.
    """
    templates = []
    lesion_type = entity["entity_type"]   # tumor or cyst
    belongs_to = entity.get("belongs_to") # e.g. kidney_1

    assert lesion_type in ["tumor", "cyst"]

    # -----------------------------
    # Priority 1: globally unique
    # -----------------------------
    if entity.get("is_only_global", False):
        add_template(
            templates,
            f"{lesion_type}_only_global",
            f"the only {lesion_type}",
            "lesion_reasoning"
        )
        return templates

    # -----------------------------
    # Priority 2: unique in kidney
    # -----------------------------
    if entity.get("is_only_in_kidney", False) and belongs_to is not None:
        add_template(
            templates,
            f"{lesion_type}_only_in_kidney",
            f"the {lesion_type} in {belongs_to}",
            "lesion_reasoning"
        )
        return templates

    # -----------------------------
    # Priority 3: kidney-level rankings
    # keep only: largest / highest / lowest
    # -----------------------------
    if belongs_to is not None:
        if entity.get("is_largest_in_kidney", False):
            add_template(
                templates,
                f"{lesion_type}_largest_in_kidney",
                f"the largest {lesion_type} in {belongs_to}",
                "lesion_reasoning"
            )

        if entity.get("is_highest_in_kidney", False):
            add_template(
                templates,
                f"{lesion_type}_highest_in_kidney",
                f"the highest {lesion_type} in {belongs_to}",
                "lesion_reasoning"
            )

        if entity.get("is_lowest_in_kidney", False):
            add_template(
                templates,
                f"{lesion_type}_lowest_in_kidney",
                f"the lowest {lesion_type} in {belongs_to}",
                "lesion_reasoning"
            )

    return templates


def attach_templates_to_case(case_data: dict):
    entities = case_data.get("entities", [])
    new_entities = []

    for entity in entities:
        e = dict(entity)

        if e.get("entity_type") == "kidney":
            e["applicable_templates"] = build_templates_for_kidney(e)

        elif e.get("entity_type") in ["tumor", "cyst"]:
            e["applicable_templates"] = build_templates_for_lesion(e)

        else:
            e["applicable_templates"] = []

        e["num_applicable_templates"] = len(e["applicable_templates"])
        new_entities.append(e)

    out_case = {
        "case_id": case_data.get("case_id"),
        "entities": new_entities,
        "_summary": case_data.get("_summary", {})
    }
    return out_case


# =========================================================
# Main
# =========================================================
def main():
    assert os.path.exists(entity_json), f"entity_json not found: {entity_json}"
    safe_mkdir_for_file(output_json)

    with open(entity_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    ok = 0
    err = 0

    case_ids = sorted([k for k in data.keys() if k.startswith("case_")])

    for case_id in case_ids:
        try:
            out[case_id] = attach_templates_to_case(data[case_id])
            ok += 1
            print(f"[OK] {case_id}")
        except Exception as e:
            err += 1
            out[case_id] = {
                "case_id": case_id,
                "_error": repr(e)
            }
            print(f"[Error] {case_id}: {repr(e)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n========== Done ==========")
    print(f"[Summary] ok={ok}, err={err}")
    print(f"[Saved] {output_json}")


if __name__ == "__main__":
    main()
