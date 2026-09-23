"""Compare one regularized epoch against its matched control; no model forward."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from metrics import classification_map
from data import CLASSES

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / 'runs/step11_lrbranch_20260922_094801'
WEAK = ('bottle', 'pottedplant', 'sofa')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--single-only', action='store_true', help='stop after a numerically ineffective treatment; no multiscale claim')
    args = parser.parse_args()
    run = args.run
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    cfg, control_cfg = read(run/'config.json'), read(CONTROL/'config.json')
    for key in ('resume_checkpoint_sha256', 'learning_rates', 'train_scales', 'batch_size', 'seed'):
        assert cfg[key] == control_cfg[key], key
    for name in ('train_ids.json', 'val_ids.json', 'epoch7_scale_assignment.json'):
        assert read(run/name) == read(CONTROL/name), name
    a = torch.load(run/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    b = torch.load(CONTROL/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    identical = a['model'].keys() == b['model'].keys() and all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    comparison = dict(model_tensors_bitwise_equal=identical, tensor_count=len(a['model']),
                      max_parameter_difference=max(float((v-b['model'][k]).abs().max()) for k,v in a['model'].items() if v.is_floating_point()),
                      decoupled_weight_decay=[g.get('decoupled_weight_decay') for g in a['optimizer']['param_groups']])
    (run/'checkpoint_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    del a, b
    if args.single_only:
        new = read(run/'result.json')['final']
        old = read(CONTROL/'result.json')['final']
        factors={k:float(np.float32(1-cfg['learning_rates'][k]*v)) for k,v in cfg['weight_decays'].items()}
        assert all(v==1 for v in factors.values()), 'Early-stop mode requires ineffective decay evidence'
        delta={k:100*(new['ap'][k]-old['ap'][k]) for k in CLASSES}
        result=dict(comparison, status='completed', promote=False, scope='single480 full VOC2012 val',
                    matched_source_config_ids_and_scale_assignment=True,
                    control_map=old['val_map'], candidate_map=new['val_map'],
                    delta_map_pp=100*(new['val_map']-old['val_map']),
                    weak_mean_delta_pp=float(np.mean([delta[k] for k in WEAK])),
                    ap_delta_pp=delta, float32_decay_factors=factors,
                    triple_evaluation_executed=False, bootstrap_executed=False,
                    reason='Numerically ineffective treatment: stop before costly multiscale inference. Tiny score differences cannot establish regularization benefit; deterministic algorithms were not enforced.',
                    next_candidate=dict(optimizer='adamw',weight_decays=[0.,0.,.01],training_executed=False,
                                        rationale='Head-only effective decay; validate real parameter shrink before another one-epoch branch.'))
        (run/'matched_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        return
    if identical:
        # Equal complete model states with identical preprocessing imply identical inference.
        # Preserve provenance instead of copying old predictions as a new forward pass.
        saved = np.load(run/'last_predictions.npz')
        old_saved = np.load(CONTROL/'last_predictions.npz')
        for key in ('image_ids', 'labels', 'scores'):
            np.testing.assert_array_equal(saved[key], old_saved[key])
        prior = read(CONTROL/'triple_evaluation/result.json')
        result = dict(comparison, status='completed', promote=False,
                      inference_executed=False, matched_source_config_ids_and_scale_assignment=True,
                      reference_predictions=str(CONTROL/'triple_evaluation/predictions.npz'),
                      triple_map_inherited_by_exact_model_identity=prior['val_map'],
                      delta_map_pp=0., weak_mean_delta_pp=0., paired_bootstrap_interval_pp=[0.,0.],
                      bootstrap_executed=False,
                      conclusion='All model tensors and single-scale predictions are bitwise identical. No effective regularization change; skip redundant multiscale inference and bootstrap.')
        (run/'matched_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    candidate = np.load(run/'triple_evaluation/predictions.npz')
    reference = np.load(CONTROL/'triple_evaluation/predictions.npz')
    for key in ('image_ids', 'labels'):
        np.testing.assert_array_equal(candidate[key], reference[key])
    y, s, s0 = candidate['labels'], candidate['scores'], reference['scores']
    new, old = classification_map(y, s), classification_map(y, s0)
    delta = (new['ap'] - old['ap']) * 100
    weak_indices = [CLASSES.index(c) for c in WEAK]
    other_indices = [i for i, c in enumerate(CLASSES) if c not in WEAK]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(500):
        ii = rng.integers(0, len(y), len(y))
        samples.append((classification_map(y[ii], s[ii])['ap'] - classification_map(y[ii], s0[ii])['ap']) * 100)
    samples = np.asarray(samples)
    np.save(run/'paired_control_bootstrap.npy', samples)
    gates = dict(overall_gain_at_least_0_10pp=bool(delta.mean() >= .10),
                 weak_mean_drop_at_most_0_10pp=bool(delta[weak_indices].mean() >= -.10),
                 other_class_drop_at_most_1pp=bool(delta[other_indices].min() >= -1.))
    train_result = read(run/'result.json')
    result = dict(control=str(CONTROL.relative_to(ROOT)), candidate=str(run.relative_to(ROOT)),
                  matched_source_config_ids_and_scale_assignment=True,
                  control_map=float(old['map']), candidate_map=float(new['map']),
                  delta_map_pp=float(delta.mean()), weak_mean_delta_pp=float(delta[weak_indices].mean()),
                  ap_delta_pp=dict(zip(CLASSES, delta.tolist())),
                  paired_bootstrap=dict(replicates=500, seed=42, map_percentile95_pp=np.percentile(samples.mean(1), [2.5, 97.5]).tolist()),
                  gates=gates, promote=all(gates.values()),
                  theoretical_decay_only_shrink={k:float(1-(1-cfg['learning_rates'][k]*v)**train_result['updates']) for k,v in cfg['weight_decays'].items()},
                  caveat='Conditional on two checkpoints and reused val; no independent test or training-seed uncertainty.')
    (run/'matched_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
