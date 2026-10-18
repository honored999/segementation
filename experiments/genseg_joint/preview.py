"""Qualitative inner61-only G preview; never a training dataset or metric.

Safe CPU deserialization includes D (and joint S) tensors; only G is constructed.
"""
import hashlib
import math
import re
from pathlib import Path
import numpy as np
from experiments.genseg30.contract import (require, read_json, write_json, sha256,
    capture_identities, assert_identities, check_output, checked_file, validate_arrays)
from .data import Cohort, inner_split
from .models import Generator, source_identity
from .runtime import audit, gpu_guard, CLASSIFICATION
from .pretrain import audit_pretrained, validate_protocol

TITLE_HEIGHT=24
CLASSES={CLASSIFICATION,'engineering_genseg_joint_pretrained_GD_fresh_UNet_not_formal'}
CLASS='engineering_genseg_joint_qualitative_preview_not_formal_not_training_dataset'


def hash_stream(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def audit_joint(target,run,digest):
    """Reject phase/provenance errors before arrays, tensors or G construction."""
    require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Explicit lowercase checkpoint SHA256 required')
    run=Path(run).resolve()
    paths=[checked_file(run,'complete.json'),checked_file(run,'joint.pt')]
    identities=capture_identities(paths)
    receipt=read_json(paths[0]); p=receipt.get('protocol')
    require(receipt.get('format')==1 and receipt.get('complete') is True and
            receipt.get('checkpoint')=='joint.pt' and receipt.get('classification') in CLASSES,
            'Completed joint receipt required')
    require(isinstance(p,dict) and p.get('classification')==receipt['classification'] and p.get('phase')=='joint',
            'Joint protocol required')
    require(type(receipt.get('steps')) is int and receipt['steps']>0 and receipt['steps']==p.get('steps'), 'Joint step mismatch')
    require(p.get('seed')==20261018,'Fixed inner61 seed required')
    require(p.get('split')==inner_split(target.train,target.val,20261018),'Fixed inner61/internal15/external19 mismatch')
    require(p.get('target_identity')==target.identity and p.get('target_manifest_sha256')==target.manifest_sha256,'Target identity mismatch')
    require(p.get('source')==source_identity(),'Pinned source identity mismatch')
    expected={'G_width':64,'G_downs':8,'D_width':64,'S':'exact_author_UNet(1,1)_width64','dtype':'float32'}
    require(p.get('models')==expected,'Joint model configuration mismatch')
    for key,value in {
        'normalization':'inner61_only_valid_pixel_minmax_minus1_plus1_no_clip',
        'pad':'native_full_planes_bottom_right_to_multiple256; image0_label-1_no_resize',
        'architecture':'upconv_only; unused_plain_conv_alpha_frozen_not_updated',
        'BN':'eval_frozen_buffers_all_virtual_and_actual_calls_deviation',
        'all_real_input':'same_3mm_masked_target_for_G_D_S_internal_validation'}.items():
        require(p.get(key)==value,'Joint protocol mismatch: '+key)
    scale=p.get('scaling',{})
    require(isinstance(scale,dict) and scale.get('method')=='inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip'
        and scale.get('fit_ids')==p['split']['train'] and
        all(type(scale.get(k)) in (int,float) and math.isfinite(scale[k]) for k in ('min','max')) and
        scale['max']>scale['min'],'Saved inner61 scaling required')
    identities_saved=p.get('implementation_files_sha256')
    require(isinstance(identities_saved,dict) and bool(identities_saved) and
        all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in identities_saved.items()),
        'Missing saved implementation identities')
    require(receipt.get('checkpoint_sha256')==digest==sha256(paths[1]),'Joint checkpoint hash mismatch')
    target.verify_identity(target.identity)
    assert_identities(identities)
    return receipt,identities


def audit_gan(target,run,digest):
    """Reuse the completed-pretrain policy audit without constructing D or S."""
    run=Path(run).resolve()
    paths=[checked_file(run,'complete.json'),checked_file(run,'gan.pt')]
    identities=capture_identities(paths)
    receipt=audit_pretrained(run,digest,expected={
        'seed':20261018,'split':inner_split(target.train,target.val,20261018),
        'target_identity':target.identity,'target_manifest_sha256':target.manifest_sha256,
        'model_config':{'G':{'width':64,'downs':8,'seed':20261018},'D_width':64}})
    protocol=receipt['protocol']
    require(type(receipt.get('steps')) is int,'Invalid pretrain receipt step count')
    require(all(type(v) in (int,float) for v in protocol['rates'].values()) and
        type(protocol['l1_weight']) in (int,float),'Invalid saved pretrain rates/L1')
    require(all(type(protocol['scaling'][k]) in (int,float) for k in ('min','max')),
        'Saved inner61 scaling required')
    saved=protocol.get('implementation_files_sha256')
    require(isinstance(saved,dict) and bool(saved) and
        all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in saved.items()),
        'Missing saved implementation identities')
    target.verify_identity(target.identity)
    assert_identities(identities)
    return receipt,identities


def load_checkpoint(run,filename,digest):
    import torch
    path=checked_file(run,filename)
    with path.open('rb') as stream:
        require(hash_stream(stream)==digest,'Preview checkpoint changed')
        stream.seek(0); payload=torch.load(stream,map_location='cpu',weights_only=True)
        stream.seek(0); require(hash_stream(stream)==digest,'Preview checkpoint changed')
    return payload


def load_joint(run,digest,receipt):
    payload=load_checkpoint(run,'joint.pt',digest)
    require(isinstance(payload,dict) and set(payload)=={'format','classification','steps','G','D','S','arch','model_config','optimizers','protocol','rng','source'},'Joint checkpoint schema mismatch')
    for key in ('format','classification','steps','protocol'):
        require(payload[key]==receipt[key],'Checkpoint/receipt mismatch: '+key)
    require(payload['source']==receipt['protocol']['source'],'Checkpoint source mismatch')
    require(payload['model_config']=={'G':{'width':64,'downs':8,'seed':20261018},
        'S':'pinned_author_UNet(1,1,bilinear=False)'},'Checkpoint model configuration mismatch')
    return payload


def load_gan(run,digest,receipt):
    payload=load_checkpoint(run,'gan.pt',digest)
    require(isinstance(payload,dict) and set(payload)=={'format','phase','classification','steps','G','D','arch','optimizers','protocol','rng'},
        'Invalid GAN checkpoint schema')
    for key in ('format','phase','classification','steps','protocol'):
        require(payload[key]==receipt[key],'Checkpoint/receipt mismatch: '+key)
    validate_protocol(payload['protocol'])
    require(payload['phase']=='gan_pretrain' and payload['protocol']['model_config']=={
        'G':{'width':64,'downs':8,'seed':20261018},'D_width':64},'GAN checkpoint phase/model configuration mismatch')
    expected_opt={k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,
        'steps':payload['steps'],'state':{}} for k,v in payload['protocol']['rates'].items()}
    require(payload['optimizers']==expected_opt,'Pretrain optimizer policy/count mismatch')
    require(isinstance(payload['rng'],dict),'Missing pretrain RNG record')
    return payload


def restore_generator(generator,payload):
    """Validate and stage every tensor before changing any G state or alpha alias."""
    import torch
    config=payload['protocol']['model_config'] if payload.get('phase')=='gan_pretrain' else payload['model_config']
    require(config['G']==generator.config,'Generator configuration mismatch')
    refs={'G':generator.net.state_dict(), 'arch':{k:getattr(generator.networks,k) for k in ('conv_arch','upconv_arch')}}
    for name,reference in refs.items():
        values=payload[name]
        require(isinstance(values,dict) and values.keys()==reference.keys(),'Generator state keys mismatch: '+name)
        for k,r in reference.items():
            v=values[k]
            require(isinstance(v,torch.Tensor) and v.layout==torch.strided and v.shape==r.shape and
                v.dtype==r.dtype and bool(torch.isfinite(v).all()),'Invalid generator tensor: '+name+'.'+k)
    staged={name:{k:payload[name][k].to(r.device).clone() for k,r in reference.items()}
        for name,reference in refs.items()}
    with torch.no_grad():
        for k,v in generator.net.state_dict().items(): v.copy_(staged['G'][k])
    generator.restore_arch(staged['arch'])
    generator.net.eval()
    for parameter in generator.net.parameters(): parameter.requires_grad_(False)


def select_planes(labels):
    require(labels.ndim==3 and np.isin(labels,[-1,0,1]).all(),'Invalid label volume')
    valid=(labels>=0).any(axis=(1,2)); positive=(labels==1).any(axis=(1,2))
    lesion=np.flatnonzero(valid & positive); empty=np.flatnonzero(valid & ~positive)
    chosen=([(int(lesion[0]),'lesion')] if len(lesion) else [])+[(int(z),'empty') for z in empty[:2]]
    missing=([] if len(lesion) else ['lesion'])+['empty_'+str(i+1) for i in range(len(empty[:2]),2)]
    if not valid.any(): missing.append('all_planes_ignore')
    return chosen,missing


def generate(generator,label,scale):
    import torch
    from experiments.genseg30.models import pad_plane
    condition=np.where(label>=0,label,0).astype(np.float32)
    condition=pad_plane(condition,256,0)
    with torch.no_grad():
        out=generator.net(torch.from_numpy(condition[None,None]).to(generator.device))
    require(tuple(out.shape)==(1,1,*condition.shape) and bool(torch.isfinite(out).all()),'Invalid generator output')
    h,w=label.shape
    native=out[0,0,:h,:w].cpu().numpy()
    native=((native+1)*.5*(scale['max']-scale['min'])+scale['min']).astype(np.float32)
    require(np.isfinite(native).all(),'Nonfinite inverse scaling')
    native[label<0]=0
    return native,list(condition.shape)


def render(path,target,label,generated,scale):
    from PIL import Image,ImageDraw
    require(target.shape==label.shape==generated.shape and target.ndim==2,'Native matching panels required')
    valid=label>=0; lo,hi=scale['min'],scale['max']
    report={}
    def gray(values,name):
        require(np.isfinite(values).all(),'Nonfinite display values')
        report[name+'_clipped_low']=int(((values<lo)&valid).sum())
        report[name+'_clipped_high']=int(((values>hi)&valid).sum())
        pixels=np.rint(np.clip((values-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        pixels[~valid]=0
        return np.repeat(pixels[:,:,None],3,axis=2)
    real=gray(target,'target'); fake=gray(generated,'generated')
    mask=np.repeat(((label==1)*255).astype(np.uint8)[:,:,None],3,axis=2)
    overlay=fake.copy(); lesion=label==1
    overlay[lesion]=np.rint(.6*np.array([255,0,0])+.4*fake[lesion]).astype(np.uint8)
    h,w=label.shape
    canvas=Image.new('RGB',(4*w,h+TITLE_HEIGHT),'black'); draw=ImageDraw.Draw(canvas)
    for i,(pixels,title) in enumerate(zip((real,mask,fake,overlay),('Real 3mm','Lesion label','Generated DWI','Generated + label'))):
        canvas.paste(Image.fromarray(pixels),(i*w,TITLE_HEIGHT)); draw.text((i*w+2,4),title,fill='white')
    with Path(path).open('xb') as stream: canvas.save(stream,format='PNG')
    report.update(window=[lo,hi],invalid_pixels=int((~valid).sum()),overlay={'color':[255,0,0],'alpha':.6,'background':.4},panel_shape=[h,w])
    return report


def run_preview(target,run,checkpoint_sha256,output,*,gpu,patient_limit=4,phase='joint'):
    require(phase in ('joint','gan_pretrain'),'Invalid preview phase')
    checkpoint_file='joint.pt' if phase=='joint' else 'gan.pt'
    require(type(patient_limit) is int and 1<=patient_limit<=61,'Patient limit must be 1..61')
    run=Path(run).resolve()
    output=check_output(output,list(target.protected)+[run])
    target.require_no_unpack_writes()
    auditor=audit_joint if phase=='joint' else audit_gan
    receipt,identities=auditor(target,run,checkpoint_sha256)
    inventory={p:target.inventory(p) for p in target.train}
    source=source_identity(); implementation=sha256(__file__)
    def final_audit():
        assert_identities(identities); audit(target,inventory)
        require(source_identity()==source and sha256(__file__)==implementation,'Preview source changed')
        require(sha256(checked_file(run,checkpoint_file))==checkpoint_sha256,'Preview checkpoint changed')
    initial=gpu_guard(gpu)
    import torch
    require(not torch.cuda.is_initialized(),'GPU binding must precede CUDA initialization')
    require(torch.cuda.is_available(),'CUDA-enabled PyTorch required')
    torch.cuda.set_per_process_memory_fraction(initial['allocator_fraction'],0)
    loader=load_joint if phase=='joint' else load_gan
    payload=loader(run,checkpoint_sha256,receipt)
    final_audit()
    generator=Generator(seed=20261018,device='cuda:0')
    restore_generator(generator,payload)
    del payload  # D (and joint S) tensors stayed on CPU; no D/S models instantiated.
    cohort=Cohort(target,20261018)
    require(cohort.split==receipt['protocol']['split'],'Cohort changed')
    scale=receipt['protocol']['scaling']
    selected=sorted(cohort.train)[:patient_limit]
    output.mkdir(parents=True,exist_ok=False)
    records=[]; patients=[]
    for patient in selected:
        final_audit()
        image,label,_=cohort.load('train',patient)
        validate_arrays(image,label)
        planes,missing=select_planes(label[0]); patients.append({'patient':patient,'missing_categories':missing,
            'all_ignore_planes':np.flatnonzero(~(label[0]>=0).any(axis=(1,2))).tolist()})
        for z,category in planes:
            generated,pad=generate(generator,label[0,z],scale)
            filename=f'{patient}_z{z:04d}_{category}.png'
            display=render(output/filename,image[0,z],label[0,z],generated,scale)
            records.append({'patient':patient,'z':z,'category':category,'native_shape':list(label.shape[2:]),
                'padded_shape':pad,'file':filename,'sha256':sha256(output/filename),'display':display})
    final_audit()
    if not records:
        write_json(output/'failed.json',{'format':1,'complete':False,'classification':CLASS,
            'error':'No renderable planes in selected inner61 patients','patients':patients})
        raise ValueError('No renderable planes in selected inner61 patients; see failed.json for missing/all-ignore categories')
    for record in records: require(sha256(output/record['file'])==record['sha256'],'Preview PNG changed')
    write_json(output/'preview.json',{'format':1,'complete':True,'classification':CLASS,
        'run':str(run),'phase':phase,'checkpoint':checkpoint_file,
        'checkpoint_sha256':checkpoint_sha256,'receipt_sha256':sha256(checked_file(run,'complete.json')),
        'source':source,'target_identity':target.identity,'target_manifest_sha256':target.manifest_sha256,
        'input_inventory':inventory,'saved_protocol':receipt['protocol'],'scaling':scale,
        'preview_implementation_sha256':implementation,'patients':patients,'previews':records,
        'initial_resources':initial,
        'cpu_payload':'weights_only=True, map_location=cpu; '+('G/D/S' if phase=='joint' else 'G/D')+' tensors; only G restored on GPU',
        'D_S':('D/S' if phase=='joint' else 'D (no S in GAN checkpoint)')+' CPU payload tensors only; no D/S models or optimizers constructed'})
    print(f'PREVIEW COMPLETE: {len(records)} PNGs; output={output}',flush=True)
    return output
