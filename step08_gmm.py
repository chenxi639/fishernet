"""Fit a train-cache diagonal GMM and verify Fisher initialization; no network training."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
import time
import warnings
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
import torch
from models.fisher_layer import FisherLayer
from models.dense_fisher_encoder import DenseFisherEncoder
from data.patch_inputs import VOCPatchDataset

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/step07_sample_20260920_012954_361930')
    args=parser.parse_args()
    run=ROOT/'runs'/f'step08_gmm_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True)
    def write(name,obj):(run/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(status='running',network_trained=False,optimizer_steps=0)
    start=time.perf_counter()
    try:
        source=args.source
        prior=json.loads((source/'result.json').read_text())
        cfg=json.loads((source/'config.json').read_text())
        cache=source/'descriptors.npz'
        assert prior['status']=='passed' and cfg['split']=='train'
        assert sha(cache)==prior['cache_sha256'],'cache checksum mismatch'
        with np.load(cache,allow_pickle=False) as saved:
            x=saved['descriptors'].astype(np.float64)
            offsets=saved['offsets'].copy();ids=saved['image_ids'].tolist()
        assert x.ndim==2 and x.shape[1]==256 and np.isfinite(x).all()
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        config=dict(source=str(source),cache_sha256=prior['cache_sha256'],source_config=cfg,
                    components=32,covariance_type='diag',reg_covar=1e-6,tol=1e-3,max_iter=100,
                    n_init=1,init_params='kmeans',seed=42,threads=2,sklearn=sklearn.__version__,
                    dtype='float64',script_sha256=sha(Path(__file__)))
        write('config.json',config)
        gmm=GaussianMixture(n_components=32,covariance_type='diag',reg_covar=1e-6,
                            tol=1e-3,max_iter=100,n_init=1,init_params='kmeans',random_state=42,verbose=1)
        with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gmm.fit(x)
            posterior=gmm.predict_proba(x)
            score=float(gmm.score(x))
        fit_seconds=time.perf_counter()-start
        np.savez(run/'gmm.npz',means=gmm.means_,variances=gmm.covariances_,
                 stds=np.sqrt(gmm.covariances_),weights=gmm.weights_)
        soft=posterior.sum(0);hard=np.bincount(posterior.argmax(1),minlength=32)
        result.update(gmm_fitted=True,converged=bool(gmm.converged_),iterations=int(gmm.n_iter_),
                      lower_bound=float(gmm.lower_bound_),mean_train_log_likelihood=score,
                      lower_bounds=[float(v) for v in getattr(gmm,'lower_bounds_',[])],
                      fit_seconds=fit_seconds,warnings=[str(w.message) for w in caught],
                      soft_counts=soft.tolist(),hard_counts=hard.tolist(),weights=gmm.weights_.tolist(),
                      variance_min=float(gmm.covariances_.min()),variance_max=float(gmm.covariances_.max()),
                      near_floor_variances=int((gmm.covariances_<=1.01e-6).sum()))
        assert gmm.converged_,'GMM not converged: inspect result before initialization'
        assert np.isfinite(gmm.means_).all() and np.isfinite(gmm.covariances_).all()
        assert (gmm.covariances_>0).all() and (soft>1).all() and (hard>0).all()
        with np.load(run/'gmm.npz',allow_pickle=False) as artifact:
            mu=artifact['means'];std=artifact['stds'];weights=artifact['weights']
            np.testing.assert_allclose(std**2,artifact['variances'])
        encode=runpy.run_path(str(ROOT/'step01_fisher'))['encode']
        layer=FisherLayer(32,256).double().initialize_from_gmm(mu,std)
        errors=[]
        for i in range(2):
            patches=x[offsets[i]:offsets[i+1]]
            expected,_=encode(patches,mu,std,simplified=True)
            actual=layer(torch.from_numpy(patches)[None]).detach().numpy()[0]
            np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
            errors.append(float(np.max(np.abs(actual-expected))))
        # Traditional GMM posterior contains mixture weights and determinant;
        # validate it independently, not against the simplified Fisher posterior.
        _,traditional_posterior=encode(x[:128],mu,std,weights=weights,simplified=False)
        np.testing.assert_allclose(traditional_posterior,posterior[:128],rtol=1e-8,atol=1e-10)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model,_,meta=DenseFisherEncoder.from_bundle(Path(cfg['bundle']),device)
        assert meta['model_sha256']==cfg['model_metadata']['model_sha256']
        model.initialize_from_gmm(mu,std)
        versions={n:p._version for n,p in model.named_parameters()}
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','train')
        images=[ds[ds.ids.index(i)]['image'].to(device).requires_grad_() for i in ids[:2]]
        out=model(images)  # No debug bypass: real initialization must unlock forward.
        assert out['gmm_initialized'] and torch.isfinite(out['fisher']).all()
        out['fisher'].square().mean().backward()
        norms={}
        for name,module in [('conv',model.patch_encoder.conv),('fc67',model.patch_encoder.fc67),
                            ('embedding',model.patch_encoder.embedding),('fisher',model.fisher)]:
            grads=[p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            norms[name]=sum(float(g.norm()) for g in grads)
            assert norms[name]>0
        assert all(im.grad is not None and torch.isfinite(im.grad).all() and im.grad.abs().sum()>0 for im in images)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        torch.save({k:v.detach().cpu() for k,v in model.fisher.state_dict().items()},run/'fisher_init.pt')
        result.update(status='passed',numpy_fisher_max_abs=max(errors),
                      traditional_posterior_max_abs=float(np.max(np.abs(traditional_posterior-posterior[:128]))),
                      verification_image_ids=ids[:2],full_patch_counts=out['counts'].tolist(),
                      fisher_shape=list(out['fisher'].shape),gradient_norms=norms,
                      parameters_unchanged_after_initialization=True,
                      gmm_sha256=sha(run/'gmm.npz'),total_seconds=time.perf_counter()-start)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        write('result.json',result);print('RUN',run);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
