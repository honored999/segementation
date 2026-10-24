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

    def fit_scale(self,*,skip_all_ignore=False):
        lo,hi=math.inf,-math.inf
        for p in self.train:
            x,y,_=self.load('train',p)
            valid=y>=0
            if skip_all_ignore and not valid.any(): continue
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
            return self.plane(role,patient,z,scale,arrays=(x,y))
        raise ValueError('No valid sampled plane within 100 seeded attempts')

    def plane(self,role,patient,z,scale,*,arrays=None):
        require(role in ('train','validation') and patient in self.split[role],'Forbidden patient/role before array read')
        x,y,_=self.load(role,patient) if arrays is None else (*arrays,None)
        require(0<=z<x.shape[1] and bool((y[0,z]>=0).any()),'Invalid/all-ignore indexed plane')
        label=y[0,z]
        image=(2*(x[0,z]-scale['min'])/(scale['max']-scale['min'])-1).astype(np.float32)
        image=np.where(label>=0,image,0).astype(np.float32)
        image=pad_plane(image,256,0); label=pad_plane(label.astype(np.float32),256,-1)
        return image[None,None],label[None,None],{'role':role,'patient':patient,'z':int(z),'native_shape':list(x.shape[2:])}


class EpochSampler:
    """One scan of inner61; keep indices only, never retain patient arrays.

    Every epoch is a seeded permutation without replacement. All-ignore planes
    are excluded from the denominator. State supports strict complete-epoch recovery.
    """
    def __init__(self,cohort,rng):
        self.cohort=cohort; self.rng=rng; self.planes=[]
        for patient in sorted(cohort.train):
            x,y,_=cohort.load('train',patient)
            self.planes.extend((patient,int(z)) for z in np.flatnonzero((y[0]>=0).any(axis=(1,2))))
        require(bool(self.planes),'No valid inner61 planes')
        self.order=rng.permutation(len(self.planes)).tolist(); self.position=0; self.steps=0

    def next(self,scale):
        if self.position==len(self.planes):
            self.order=self.rng.permutation(len(self.planes)).tolist(); self.position=0
        patient,z=self.planes[self.order[self.position]]
        result=self.cohort.plane('train',patient,z,scale)
        self.position+=1; self.steps+=1
        return result

    def progress(self):
        total=len(self.planes)
        return {'actual_steps':self.steps,'total_valid_planes':total,'completed_epochs':self.steps//total,
            'epoch_progress':{'epoch':max(1,(self.steps-1)//total+1),'planes':self.position,'total':total},
            'unique_planes_seen':min(self.steps,total),'unique_coverage':min(self.steps,total)/total}

    def state_dict(self):
        return {'format':1,'planes':[[p,z] for p,z in self.planes],'order':list(self.order),
            'position':self.position,'steps':self.steps,'rng':self.rng.bit_generator.state,
            'progress':self.progress()}

    def load_state_dict(self,state):
        from copy import deepcopy
        require(set(state)=={'format','planes','order','position','steps','rng','progress'} and state['format']==1,
            'Epoch sampler schema mismatch')
        total=len(self.planes)
        require(state['planes']==[[p,z] for p,z in self.planes] and
            all(type(v) is int for v in state['order']) and sorted(state['order'])==list(range(total)) and
            state['position']==total and type(state['steps']) is int and state['steps']>0 and state['steps']%total==0,
            'Epoch sampler inventory/permutation/boundary mismatch')
        self.rng.bit_generator.state=deepcopy(state['rng'])
        self.order=list(state['order']); self.position=state['position']; self.steps=state['steps']
        require(state['progress']==self.progress(),'Epoch sampler progress mismatch')
