"""Verify the planned Adam-to-AdamW resume path without model forward or training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import torch
from step09_pilot import make_optimizer,scale_learning_rates,set_group_weight_decays

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/step12_scale_20260921_130318/best.pt'
GROUPS=('backbone','fisher','head')
DECAYS=(1e-5,1e-4,1e-4)

def main():
    run=ROOT/'runs'/f'step18_regularization_prepare_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    ck=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    source_groups=ck['optimizer']['param_groups']
    # Scalar placeholders are sufficient to validate state/group restoration; no step is taken.
    params=[[torch.nn.Parameter(torch.zeros(())) for _ in group['params']] for group in source_groups]
    optimizer=make_optimizer([dict(params=p,lr=1.) for p in params],'adamw')
    optimizer.load_state_dict(ck['optimizer'])
    before,after=scale_learning_rates(optimizer,.5)
    applied=set_group_weight_decays(optimizer,DECAYS)
    steps=[]
    for state in optimizer.state.values():
        if 'step' in state:steps.append(float(state['step']))
    digest=hashlib.sha256()
    with SOURCE.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    result={
        'status':'passed',
        'training_executed':False,
        'model_forward_calls':0,
        'optimizer_steps':0,
        'source_checkpoint':str(SOURCE.relative_to(ROOT)),
        'source_checkpoint_sha256':digest.hexdigest(),
        'source_epoch':ck['epoch'],
        'source_optimizer':ck['config'].get('optimizer','Adam'),
        'planned_optimizer':'AdamW',
        'group_order':GROUPS,
        'restored_parameter_counts':[len(v) for v in params],
        'restored_state_entries':len(optimizer.state),
        'restored_step_min':min(steps),
        'restored_step_max':max(steps),
        'learning_rates_before_scale':before,
        'learning_rates_after_scale':after,
        'weight_decays':dict(zip(GROUPS,applied)),
        'checks':{
            'three_parameter_groups':len(optimizer.param_groups)==3,
            'all_states_have_same_positive_training_step':min(steps)==max(steps) and min(steps)>0,
            'half_learning_rate_applied':all(a==b*.5 for a,b in zip(after,before)),
            'decays_applied_in_declared_order':tuple(applied)==DECAYS
        },
        'limitation':'This validates configuration and optimizer-state compatibility only; no loss, gradient, prediction, or accuracy was computed.'
    }
    if not all(result['checks'].values()):raise RuntimeError(result)
    (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (run/'step18_verify_regularization.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print(run)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
