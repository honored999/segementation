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


def test_split_roles_rejected_before_reads_scaling_inner_only():
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


def test_bad_split_rejected_before_dataset_creation():
    with pytest.raises(ValueError): inner_split(['a']*76,['z']*19,1)
    with pytest.raises(ValueError): inner_split(Target.train,Target.train[:19],1)
