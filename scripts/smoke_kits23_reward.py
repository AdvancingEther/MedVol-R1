"""Exercise mask decoding and scoring using a tiny synthetic sparse CT volume."""
import importlib.util
import json
import math
import tempfile
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]

def load_reward(path):
    spec = importlib.util.spec_from_file_location('kits23_reward_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module):
    with tempfile.TemporaryDirectory(prefix='medvol-smoke-') as temp:
        case = Path(temp) / 'synthetic_case'
        case.mkdir()
        # Source convention: sparse (C,H*W*Z), idx=(y*W+x)*Z+z.
        z_count = 3
        columns = [(y * 512 + x) * z_count + 1 for y in range(52, 104) for x in range(52, 104)]
        mask = sparse.csr_matrix((np.ones(len(columns)), (np.zeros(len(columns), dtype=int), columns)),
                                 shape=(14, 512 * 512 * z_count))
        sparse.save_npz(case / 'mask_(14,512,512,3).npz', mask)
        module.KITS23_NPY_ROOT = temp
        response = '<think>synthetic check</think><answer>[{"slice":1,"bbox_2d_list":[[100,100,200,200]]}]</answer>'
        boxes, _ = module._parse_pred_bboxes_and_slice(response)
        gt = dict(mask_rel_path='synthetic_case/mask_(14,512,512,3).npz',
                  template_index=0, gt_bbox_2d_list_512=boxes)
        inputs = [dict(response=response, ground_truth=gt),
                  dict(response='invalid response', ground_truth=gt),
                  dict(response=response.replace('"slice":1', '"slice":99'), ground_truth=gt)]
        scores = module.compute_score(inputs)
        assert math.isclose(scores[0]['overall'], 1.0), scores
        assert scores[1]['overall'] == 0.0, scores
        assert scores[2]['time'] == 0.0, scores
        assert all(0.0 <= value <= 1.0 for row in scores for value in row.values())
        return scores


def main():
    path = ROOT / 'third_party/EasyR1/examples/reward_function/kits23.py'
    print(json.dumps(dict(status='passed', scores=run(load_reward(path))), indent=2))

if __name__ == '__main__':
    main()
