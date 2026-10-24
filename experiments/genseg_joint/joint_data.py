"""Private immutable input snapshot and independent positive-mask sampler."""
from pathlib import Path
import time
import numpy as np
from experiments.genseg30.contract import require, sha256
from .data import Cohort
from .runtime import audit, implementation_identity


def stat_identity(path):
    p = Path(path).resolve(strict=True)
    s = p.stat()
    return (str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def cache_plan(target, patients, loader_workspace_bytes):
    """Header-only audit; external codec/pickle workspace is an explicit reserve, not a peak proof."""
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    import blosc2
    require(type(loader_workspace_bytes) is int and loader_workspace_bytes > 0,
            'Explicit positive loader workspace reserve required')
    original = getattr(target, 'original', None)
    require(original is not None and original.dataset_class is nnUNetDatasetBlosc2,
            'Unaudited backend: joint snapshot supports Blosc2 only; npz/npy refused')
    plans = {}
    for patient in patients:
        stores = [blosc2.open(urlpath=str(Path(original.folder)/(patient+suffix)), mode='r',
                             dparams={'nthreads':1}) for suffix in ('.b2nd','_seg.b2nd')]
        shape = tuple(target.manifest['cases'][patient]['source_shape'])
        require(len(shape)==4 and shape[0]==1 and all(type(v) is int and v>0 for v in shape),
                'Invalid snapshot shape metadata')
        dx,dy = [np.dtype(store.dtype) for store in stores]
        require(all(tuple(store.shape)==shape for store in stores), 'Source header shape mismatch')
        require(dx.kind=='f' and dx.itemsize<=8 and dy.kind=='i' and dy.itemsize<=8,
                'Unaudited source dtype; floating image/signed label up to 8 bytes required')
        del stores
        # Mask NPZ is inspected through its NPY header only, never decompressed as an array.
        import zipfile
        with zipfile.ZipFile(target._case_paths[patient]['mask_file']) as archive:
            require(archive.namelist()==['keep.npy'], 'Unexpected mask metadata members')
            with archive.open('keep.npy') as stream:
                version=np.lib.format.read_magic(stream)
                require(version in ((1,0),(2,0)), 'Unaudited mask NPY header version')
                mask_shape,_,mask_dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0)
                    else np.lib.format.read_array_header_2_0(stream))
        require(tuple(mask_shape)==shape[1:] and np.dtype(mask_dtype)==np.dtype(bool), 'Mask header mismatch')
        n=1
        for dim in shape: n*=dim
        final=n*(4+dy.itemsize)
        # Conservative simultaneous envelope: source x/y + Target x copy + keep/~keep,
        # Source finite/isin (comparison/table/index work up to 32 bytes/voxel), and
        # snapshot validation work. Final owned x/y already counted separately.
        transient=n*(2*dx.itemsize+dy.itemsize+2+32+32)+loader_workspace_bytes
        plans[patient]=dict(shape=shape,image_dtype=dx.str,label_dtype=dy.str,
            final_bytes=final,load_transient_bytes=transient)
    return dict(cases=plans,final_cache_estimated_bytes=sum(p['final_bytes'] for p in plans.values()),
        max_case_load_transient_bytes=max(p['load_transient_bytes'] for p in plans.values()),
        loader_workspace_reserve_bytes=loader_workspace_bytes,
        external_loader_workspace='UNKNOWN; explicit reserve, not a measured upper bound')


class InputSnapshot:
    def __init__(self, cohort, *, budget_bytes, resource_guard, loader_workspace_bytes):
        require(type(budget_bytes) is int and budget_bytes > 0, 'Explicit positive RAM byte budget required')
        self.target = cohort.target
        self.inventory = {p:self.target.inventory(p) for p in self.target.train}
        self.all_stats = {str(p):stat_identity(p) for p in self.target._identities}
        self.paths = [Path(p) for p in self.target._identities if Path(p).is_file()]
        require(bool(self.paths), 'Snapshot requires bound input paths')
        self.stats = {str(p):stat_identity(p) for p in self.paths}
        self.hashes = {str(p):sha256(p) for p in self.paths}
        self.code = implementation_identity()
        self.timings = {}
        self.arrays = {}
        self.nbytes = 0
        started = time.perf_counter()
        self.full_audit()
        self.timings['startup_full_audit'] = time.perf_counter()-started
        self.plan = cache_plan(self.target,cohort.train+cohort.val,loader_workspace_bytes)
        self.final_cache_estimated_bytes = self.plan['final_cache_estimated_bytes']
        self.max_case_load_transient_bytes = self.plan['max_case_load_transient_bytes']
        self.estimated_bytes = self.final_cache_estimated_bytes
        require(self.estimated_bytes <= budget_bytes, 'Snapshot RAM estimate exceeds explicit budget')
        started = time.perf_counter()
        for role in ('train','validation'):
            for patient in cohort.split[role]:
                plan = self.plan['cases'][patient]
                resource_guard(self.estimated_bytes-self.nbytes+plan['load_transient_bytes'])
                before = {str(p):(stat_identity(p),sha256(p)) for p in self.paths_for(patient)}
                x,y,_ = cohort.load(role,patient)
                require(tuple(x.shape)==plan['shape'] and tuple(y.shape)==plan['shape'] and
                        x.dtype.str==plan['image_dtype'] and y.dtype.str==plan['label_dtype'],
                        'Loaded arrays disagree with audited headers')
                require(x.nbytes+y.nbytes == int(np.prod(plan['shape']))*(x.dtype.itemsize+y.dtype.itemsize),
                        'Loaded byte count mismatch')
                x = np.array(x,dtype=np.float32,order='C',copy=True)
                y = np.array(y,order='C',copy=True)
                require(x.shape == y.shape and np.isfinite(x).all() and np.isin(y,[-1,0,1]).all(),
                        'Invalid snapshot arrays')
                require(x.nbytes+y.nbytes==plan['final_bytes'], 'Owned cache estimate mismatch')
                self.nbytes += x.nbytes+y.nbytes
                require(self.nbytes <= budget_bytes, 'Snapshot actual RAM exceeds explicit budget')
                require(before == {str(p):(stat_identity(p),sha256(p)) for p in self.paths_for(patient)},
                        'Source changed while loading snapshot')
                x.setflags(write=False); y.setflags(write=False)
                self.arrays[patient] = (x,y)
                del x,y
                resource_guard(0)
        self.timings['cache_load'] = time.perf_counter()-started
        self.full_audit()
        self.cohort = Cohort(self,20261018,dataset=object())
        require(self.cohort.split == cohort.split, 'Snapshot split mismatch')

    @property
    def train(self): return self.target.train

    @property
    def val(self): return self.target.val

    def paths_for(self,patient):
        allowed = set(self.target._paths_for(patient))
        return [p for p in self.paths if str(p) in allowed]

    def load_train(self,dataset,patient):
        require(patient in self.arrays, 'Forbidden snapshot patient')
        x,y = self.arrays[patient]
        # Expose read-only views, not the owned arrays whose WRITEABLE can be re-enabled.
        return x.view(),y.view(),{}

    def lightweight_audit(self):
        require(self.all_stats == {str(p):stat_identity(p) for p in self.all_stats}, 'Input stat identity changed')
        require(self.code == implementation_identity(), 'Implementation changed during run')

    def full_audit(self):
        started = time.perf_counter()
        self.lightweight_audit()
        audit(self.target,self.inventory)
        require(self.hashes == {str(p):sha256(p) for p in self.paths}, 'Input content hash changed')
        self.timings['last_full_audit'] = time.perf_counter()-started


class PositiveSampler:
    def __init__(self,cohort,seed=20261018):
        self.cohort = cohort
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x504F53]))
        self.planes = []
        self.steps = 0
        for patient in sorted(cohort.train):
            _,y,_ = cohort.load('train',patient)
            self.planes.extend((patient,int(z)) for z in np.flatnonzero((y[0] == 1).any(axis=(1,2))))
        require(bool(self.planes), 'No positive inner61 planes; no retry/fallback')

    def next(self,scale):
        patient,z = self.planes[int(self.rng.integers(len(self.planes)))]
        _,label,record = self.cohort.plane('train',patient,z,scale)
        fy,fx = bool(self.rng.integers(2)),bool(self.rng.integers(2))
        if fy: label = np.flip(label,-2)
        if fx: label = np.flip(label,-1)
        label = np.array(label,order='C',copy=True)
        foreground = int((label == 1).sum())
        require(foreground > 0, 'Positive augmentation lost lesion')
        self.steps += 1
        return label,dict(record,category='positive',foreground_pixels=foreground,flip_y=fy,flip_x=fx)

    def state_dict(self):
        return dict(steps=self.steps,positive_planes=len(self.planes),planes=[[p,z] for p,z in self.planes],rng=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        from copy import deepcopy
        require(set(state)=={'steps','positive_planes','planes','rng'} and
            state['planes']==[[p,z] for p,z in self.planes] and state['positive_planes']==len(self.planes) and
            type(state['steps']) is int and state['steps']>0,'Positive sampler inventory/count mismatch')
        self.rng.bit_generator.state=deepcopy(state['rng']); self.steps=state['steps']
