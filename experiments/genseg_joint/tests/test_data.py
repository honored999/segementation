from pathlib import Path
import numpy as np
import pytest
from experiments.genseg_joint.data import Cohort, inner_split


class Target:
    train=[f't{i:02}' for i in range(76)]
    val=[f'e{i:02}' for i in range(19)]
    def __init__(self): self.reads=[]
    def load_train(self, dataset, patient):
        self.reads.append(patient)
        x=np.ones((1,1,17,19),np.float32)*(int(patient[1:])+1)
        y=np.zeros_like(x,dtype=np.int16)
        return x,y,{}


def test_split_roles_rejected_before_reads_scaling_inner_only(monkeypatch):
    target=Target(); cohort=Cohort(target,123,dataset=object())
    assert tuple(map(len,(cohort.train,cohort.val,cohort.external)))==(61,15,19)
    assert inner_split(target.train,target.val,123)==inner_split(list(reversed(target.train)),target.val,123)
    for role,patient in [('train',cohort.val[0]),('validation',cohort.train[0]),('train',cohort.external[0])]:
        with pytest.raises(ValueError,match='role'):
            cohort.load(role,patient)
    assert not target.reads
    scale=cohort.fit_scale()
    assert set(target.reads)==set(cohort.train)
    assert scale['max']==max(int(p[1:])+1 for p in cohort.train)
    image,label,record=cohort.sample('validation',np.random.default_rng(1),scale)
    assert image.shape==label.shape==(1,1,256,256)
    assert (label[:,:,17:]==-1).all()
    assert record['role']=='validation'

    from experiments.genseg_joint.data import EpochSampler
    original=target.load_train
    def planes(*args):
        x,y,meta=original(*args)
        x=np.repeat(x,3,axis=1); y=np.repeat(y,3,axis=1); y[:,0]=-1
        return x,y,meta
    monkeypatch.setattr(target,'load_train',planes)
    rng=np.random.default_rng(18); sampler=EpochSampler(cohort,rng)
    other=EpochSampler(cohort,np.random.default_rng(18))
    assert len(sampler.planes)==122 and not hasattr(sampler,'arrays')
    target.reads.clear(); records=[]
    for _ in range(122):
        _,label,record=sampler.next(scale); records.append((record['patient'],record['z']))
        assert record==other.next(scale)[2] and (label[:,:,17:]==-1).all()
    assert len(target.reads)==244  # one patient load per next(), no queue rescan
    assert len(set(records))==122 and set(records)==set(sampler.planes)
    assert sampler.progress()['completed_epochs']==1 and sampler.progress()['unique_coverage']==1.
    sampler.next(scale)
    assert sampler.progress()['completed_epochs']==1 and sampler.progress()['epoch_progress']['planes']==1
    assert sampler.state_dict()['rng']==rng.bit_generator.state


def test_bad_split_rejected_before_dataset_creation():
    with pytest.raises(ValueError): inner_split(['a']*76,['z']*19,1)
    with pytest.raises(ValueError): inner_split(Target.train,Target.train[:19],1)
