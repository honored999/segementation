"""Patient-level nested roles checked BEFORE the broader 76-case adapter reads."""
import math
import numpy as np
from experiments.genseg30.contract import require
from experiments.genseg30.models import pad_plane


def inner_split(train,external,seed):
    require(len(train)==len(set(train))==76 and len(external)==len(set(external))==19 and
            not set(train)&set(external),'Fixed disjoint outer76/19 required')
    order=np.random.default_rng(seed).permutation(sorted(train)).tolist()
    return {'train':sorted(order[:61]),'validation':sorted(order[61:]),'external':sorted(external)}


class Cohort:
    def __init__(self,target,seed,*,dataset=None):
        self.target=target
        self.split=inner_split(target.train,target.val,seed)
        self.train=self.split['train']; self.val=self.split['validation']; self.external=self.split['external']
        self.dataset=target.training_dataset() if dataset is None else dataset

    def load(self,role,patient):
        require(role in ('train','validation') and patient in self.split[role], 'Forbidden patient/role before array read')
        return self.target.load_train(self.dataset,patient)

    def fit_scale(self):
        lo,hi=math.inf,-math.inf
        for p in self.train:
            x,y,_=self.load('train',p)
            valid=y>=0
            require(bool(valid.any()),'No valid train pixels')
            lo=min(lo,float(x[valid].min())); hi=max(hi,float(x[valid].max()))
        require(math.isfinite(lo) and math.isfinite(hi) and hi>lo,'Degenerate inner61 scaling')
        return {'method':'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip','min':lo,'max':hi,
                'fit_ids':self.train}

    def sample(self,role,rng,scale):
        require(role in ('train','validation'),'Forbidden sampling role')
        # Bounded attempt policy; no hidden label-conditioned lesion oversampling.
        for _ in range(100):
            patient=str(rng.choice(self.split[role])); x,y,_=self.load(role,patient)
            z=int(rng.integers(x.shape[1])); label=y[0,z]
            if not (label>=0).any(): continue
            image=(2*(x[0,z]-scale['min'])/(scale['max']-scale['min'])-1).astype(np.float32)
            image=np.where(label>=0,image,0).astype(np.float32)
            image=pad_plane(image,256,0)
            label=pad_plane(label.astype(np.float32),256,-1)
            return image[None,None],label[None,None],{'role':role,'patient':patient,'z':z,'native_shape':list(x.shape[2:])}
        raise ValueError('No valid sampled plane within 100 seeded attempts')
