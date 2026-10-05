#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert info_stats.json into a standardized entity-dictionary format.

Input:
    info_json

Output:
    entity_json

For each case:
    {
      "case_id": "...",
      "entities": [
          {... kidney entity ...},
          {... tumor entity ...},
          {... cyst entity ...}
      ],
      "_summary": {...}
    }

No argparse. Edit paths below.
"""

import json
import os
from pathlib import Path


# =========================================================
# Global config
# =========================================================
info_json = os.environ.get('MEDVOL_STEP3_GEN_ATTRI_STATS_INFO_JSON', 'data/kits23/final_vqa_gen/info_stats.json')
entity_json = os.environ.get('MEDVOL_STEP3_GEN_ATTRI_STATS_ENTITY_JSON', 'data/kits23/final_vqa_gen/entity_dict.json')


# =========================================================
# Helpers
# =========================================================
def safe_mkdir_for_file(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sort_by_voxels_desc(items):
    return sorted(items, key=lambda x: (-x.get("voxels", 0), x.get("instance_id", 10**9)))


def sort_by_height_asc(items):
    # smaller relative_height = more upper
    return sorted(items, key=lambda x: (x.get("relative_height", 1e9), x.get("instance_id", 10**9)))


# =========================================================
# Core
# =========================================================
def build_case_entity_dict(case_id: str, case_data: dict) -> dict:
    """
    Convert one case from info_stats format to entity-dictionary format.
    """
    # -----------------------------------------------------
    # collect kidneys
    # -----------------------------------------------------
    kidney_keys = [k for k in case_data.keys() if k.startswith("kidney_")]
    kidney_keys.sort(key=lambda x: int(x.split("_")[1]))

    kidneys = []
    all_tumors = []
    all_cysts = []

    for kidney_key in kidney_keys:
        kd = case_data[kidney_key]

        tumors = kd.get("tumors", [])
        cysts = kd.get("cysts", [])

        kidneys.append({
            "entity_id": kidney_key,
            "entity_type": "kidney",
            "mask_file": kd.get("file"),
            "instance_id": kd.get("instance_id"),
            "annotation_id": kd.get("annotation_id"),
            "voxels": kd.get("voxels", 0),
            "centroid_axis0_axis1_axis2": kd.get("centroid_axis0_axis1_axis2"),
            "axis0_range": kd.get("axis0_range", [-1, -1]),
            "num_tumors": len(tumors),
            "num_cysts": len(cysts),
            "has_tumor": len(tumors) > 0,
            "has_cyst": len(cysts) > 0,
            "has_lesion": (len(tumors) + len(cysts)) > 0,
            "is_healthy": (len(tumors) + len(cysts)) == 0,
        })

        for t in tumors:
            t_copy = dict(t)
            t_copy["_belongs_to"] = kidney_key
            all_tumors.append(t_copy)

        for c in cysts:
            c_copy = dict(c)
            c_copy["_belongs_to"] = kidney_key
            all_cysts.append(c_copy)

    # -----------------------------------------------------
    # derive kidney-level relational attributes
    # -----------------------------------------------------
    num_kidneys_with_tumor = sum(1 for k in kidneys if k["has_tumor"])
    num_kidneys_with_cyst = sum(1 for k in kidneys if k["has_cyst"])
    num_kidneys_with_lesion = sum(1 for k in kidneys if k["has_lesion"])
    num_healthy_kidneys = sum(1 for k in kidneys if k["is_healthy"])

    # optional comparison: largest tumor burden / cyst burden by count
    if len(kidneys) > 0:
        max_tumor_count = max(k["num_tumors"] for k in kidneys)
        max_cyst_count = max(k["num_cysts"] for k in kidneys)
    else:
        max_tumor_count = 0
        max_cyst_count = 0

    num_kidneys_max_tumor_count = sum(1 for k in kidneys if k["num_tumors"] == max_tumor_count)
    num_kidneys_max_cyst_count = sum(1 for k in kidneys if k["num_cysts"] == max_cyst_count)
    for k in kidneys:
        k["has_both_tumor_and_cyst"] = k["has_tumor"] and k["has_cyst"]

    num_kidneys_with_both_tumor_and_cyst = sum(
        1 for k in kidneys if k["has_both_tumor_and_cyst"]
    )

    # 再统一计算依赖这些字段的属性
    for k in kidneys:
        k["is_only_kidney_with_tumor"] = k["has_tumor"] and (num_kidneys_with_tumor == 1)
        k["is_only_kidney_with_cyst"] = k["has_cyst"] and (num_kidneys_with_cyst == 1)
        k["is_only_kidney_with_lesion"] = k["has_lesion"] and (num_kidneys_with_lesion == 1)
        k["is_only_healthy_kidney"] = k["is_healthy"] and (num_healthy_kidneys == 1)

        k["is_only_kidney_with_both_tumor_and_cyst"] = (
            k["has_both_tumor_and_cyst"] and
            (num_kidneys_with_both_tumor_and_cyst == 1)
        )

        k["has_most_tumors"] = (
            k["num_tumors"] > 0 and
            k["num_tumors"] == max_tumor_count and
            num_kidneys_max_tumor_count == 1
        )

        k["has_most_cysts"] = (
            k["num_cysts"] > 0 and
            k["num_cysts"] == max_cyst_count and
            num_kidneys_max_cyst_count == 1
        )

    # -----------------------------------------------------
    # build lesion entities (tumor / cyst)
    # -----------------------------------------------------
    lesion_entities = []

    def enrich_lesions(lesions, lesion_type: str):
        """
        lesions: list of raw lesion dicts, each has _belongs_to
        """
        if len(lesions) == 0:
            return []

        # global rankings
        by_size_global = sort_by_voxels_desc(lesions)
        by_height_global = sort_by_height_asc(lesions)

        largest_global_inst_id = by_size_global[0].get("instance_id")
        smallest_global_inst_id = by_size_global[-1].get("instance_id")
        highest_global_inst_id = by_height_global[0].get("instance_id")
        lowest_global_inst_id = by_height_global[-1].get("instance_id")

        # uniqueness checks
        if len(by_size_global) >= 2:
            is_largest_unique_global = by_size_global[0].get("voxels", 0) > by_size_global[1].get("voxels", 0)
            is_smallest_unique_global = by_size_global[-1].get("voxels", 0) < by_size_global[-2].get("voxels", 0)
        else:
            is_largest_unique_global = True
            is_smallest_unique_global = True

        if len(by_height_global) >= 2:
            is_highest_unique_global = by_height_global[0].get("relative_height", 1e9) < by_height_global[1].get("relative_height", 1e9)
            is_lowest_unique_global = by_height_global[-1].get("relative_height", -1e9) > by_height_global[-2].get("relative_height", -1e9)
        else:
            is_highest_unique_global = True
            is_lowest_unique_global = True

        out = []

        for lesion in lesions:
            belongs_to = lesion["_belongs_to"]
            inst_id = lesion.get("instance_id")

            # collect same-type lesions in the same kidney
            same_kidney_lesions = [x for x in lesions if x["_belongs_to"] == belongs_to]

            by_size_kidney = sort_by_voxels_desc(same_kidney_lesions)
            by_height_kidney = sort_by_height_asc(same_kidney_lesions)

            largest_kidney_inst_id = by_size_kidney[0].get("instance_id")
            smallest_kidney_inst_id = by_size_kidney[-1].get("instance_id")
            highest_kidney_inst_id = by_height_kidney[0].get("instance_id")
            lowest_kidney_inst_id = by_height_kidney[-1].get("instance_id")

            if len(by_size_kidney) >= 2:
                is_largest_unique_kidney = by_size_kidney[0].get("voxels", 0) > by_size_kidney[1].get("voxels", 0)
                is_smallest_unique_kidney = by_size_kidney[-1].get("voxels", 0) < by_size_kidney[-2].get("voxels", 0)
            else:
                is_largest_unique_kidney = True
                is_smallest_unique_kidney = True

            if len(by_height_kidney) >= 2:
                is_highest_unique_kidney = by_height_kidney[0].get("relative_height", 1e9) < by_height_kidney[1].get("relative_height", 1e9)
                is_lowest_unique_kidney = by_height_kidney[-1].get("relative_height", -1e9) > by_height_kidney[-2].get("relative_height", -1e9)
            else:
                is_highest_unique_kidney = True
                is_lowest_unique_kidney = True

            entity = {
                "entity_id": f"{lesion_type}_{inst_id}",
                "entity_type": lesion_type,
                "mask_file": lesion.get("file"),
                "instance_id": lesion.get("instance_id"),
                "annotation_id": lesion.get("annotation_id"),
                "voxels": lesion.get("voxels", 0),
                "centroid_axis0_axis1_axis2": lesion.get("centroid_axis0_axis1_axis2"),
                "bbox_axis0_axis1_axis2": lesion.get("bbox_axis0_axis1_axis2"),
                "belongs_to": belongs_to,
                "assignment_method": lesion.get("assignment_method"),
                "overlap_with_assigned_kidney": lesion.get("overlap_with_assigned_kidney", 0),
                "relative_height": lesion.get("relative_height", -1.0),

                # global attributes
                "is_only_global": len(lesions) == 1,
                "is_largest_global": (inst_id == largest_global_inst_id) and is_largest_unique_global,
                "is_smallest_global": (inst_id == smallest_global_inst_id) and is_smallest_unique_global,
                "is_highest_global": (inst_id == highest_global_inst_id) and is_highest_unique_global,
                "is_lowest_global": (inst_id == lowest_global_inst_id) and is_lowest_unique_global,

                # kidney-level attributes
                "is_only_in_kidney": len(same_kidney_lesions) == 1,
                "is_largest_in_kidney": (inst_id == largest_kidney_inst_id) and is_largest_unique_kidney,
                "is_smallest_in_kidney": (inst_id == smallest_kidney_inst_id) and is_smallest_unique_kidney,
                "is_highest_in_kidney": (inst_id == highest_kidney_inst_id) and is_highest_unique_kidney,
                "is_lowest_in_kidney": (inst_id == lowest_kidney_inst_id) and is_lowest_unique_kidney,

                "num_same_type_in_kidney": len(same_kidney_lesions),
                "num_same_type_global": len(lesions),
            }
            out.append(entity)

        return out

    lesion_entities.extend(enrich_lesions(all_tumors, "tumor"))
    lesion_entities.extend(enrich_lesions(all_cysts, "cyst"))

    # -----------------------------------------------------
    # final entities list
    # -----------------------------------------------------
    entities = []
    entities.extend(kidneys)
    entities.extend(lesion_entities)

    # stable sort: kidneys first, then tumor, then cyst
    type_order = {"kidney": 0, "tumor": 1, "cyst": 2}
    entities.sort(key=lambda x: (type_order.get(x["entity_type"], 99), x.get("instance_id", 10**9)))

    out_case = {
        "case_id": case_id,
        "entities": entities,
        "_summary": case_data.get("_summary", {})
    }

    return out_case


def main():
    assert os.path.exists(info_json), f"info_json not found: {info_json}"
    safe_mkdir_for_file(entity_json)

    with open(info_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    ok = 0
    err = 0

    case_ids = sorted([k for k in data.keys() if k.startswith("case_")])

    for case_id in case_ids:
        try:
            out[case_id] = build_case_entity_dict(case_id, data[case_id])
            ok += 1
            print(f"[OK] {case_id}")
        except Exception as e:
            err += 1
            out[case_id] = {
                "case_id": case_id,
                "_error": repr(e)
            }
            print(f"[Error] {case_id}: {repr(e)}")

    with open(entity_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n========== Done ==========")
    print(f"[Summary] ok={ok}, err={err}")
    print(f"[Saved] {entity_json}")


if __name__ == "__main__":
    main()
