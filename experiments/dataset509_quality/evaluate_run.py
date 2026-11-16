"""Owned full-volume best-checkpoint evaluation; historical seven metrics unchanged."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from quality import read,sha,write_new,need
from run_layout import job_layout


def detection_stats(pred,gt,voxel_ml):
    import numpy as np
    from scipy import ndimage
    need(pred.shape==gt.shape and pred.ndim==3,'MASK_SHAPE_INVALID')
    structure=np.ones((3,3,3),dtype=np.uint8)
    labels,n_gt=ndimage.label(gt,structure)
    predicted,n_pred=ndimage.label(pred,structure)
    hit_gt=np.unique(labels[pred]);hit_gt=hit_gt[hit_gt>0]
    hit_pred=np.unique(predicted[gt]);hit_pred=hit_pred[hit_pred>0]
    sizes=np.bincount(labels.ravel(),minlength=n_gt+1)[1:]*voxel_ml
    edges=[0,.1,1,10,float('inf')]
    groups=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        indices=np.flatnonzero((sizes>=lo)&(sizes<hi))+1
        groups.append({'gt_lesions':len(indices),'detected_lesions':int(np.isin(indices,hit_gt).sum())})
    return {'gt_lesions':n_gt,'detected_lesions':len(hit_gt),'false_positive_lesions':n_pred-len(hit_pred),'groups':groups}


def supplement(predictions,gt_dir,case_rows,output):
    import numpy as np
    import SimpleITK as sitk
    details=[]
    for row in case_rows:
        cid=row['case_id'];prediction=sitk.ReadImage(str(predictions/(cid+'.nii.gz')));truth=sitk.ReadImage(str(gt_dir/(cid+'.nii.gz')))
        p=sitk.GetArrayFromImage(prediction)>0;g=sitk.GetArrayFromImage(truth)>0
        stats=detection_stats(p,g,float(np.prod(truth.GetSpacing()))/1000)
        tp,fp=int(row['tp']),int(row['fp'])
        stats.update(precision=tp/(tp+fp) if tp+fp else None,
                     lesion_recall=stats['detected_lesions']/stats['gt_lesions'] if stats['gt_lesions'] else None)
        details.append(stats)
    def average(key):
        values=[float(r[key]) for r in details if r[key] is not None]
        return {'mean':float(np.mean(values)) if values else None,'valid_cases':len(values)}
    groups=[]
    for i,name in enumerate(('[0,0.1) mL','[0.1,1) mL','[1,10) mL','[10,infinity) mL')):
        count=sum(r['groups'][i]['gt_lesions'] for r in details);hit=sum(r['groups'][i]['detected_lesions'] for r in details)
        groups.append({'bin':name,'gt_lesions':count,'detected_lesions':hit,'lesion_recall':hit/count if count else None})
    result={'n_cases':len(details),'precision':average('precision'),'lesion_recall_macro':average('lesion_recall'),
            'false_positive_lesions_per_case':average('false_positive_lesions'),'lesion_volume_groups':groups,
            'detection_definition':'3D 26-connectivity; a GT lesion is detected by any positive overlap; a predicted component with no GT overlap is false positive; no one-to-one matching or minimum size removal.',
            'empty_policy':'precision undefined for empty prediction; lesion recall undefined for empty GT; null excluded from macro means with valid_cases recorded; FP count always defined.',
            'volume_bins':'Fixed before Q outcome inspection; native spacing in mm, lesion volumes in mL.',
            'historical_f2_field':'paper formula 5TP/(5TP+4FP+FN), equivalent to conventional F0.5; historical files unchanged.'}
    write_new(output,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args()
    root=Path(a.run_root);spec=read(root/'job_spec.json');layout=job_layout(root,spec)
    training=layout['training'];c=read(layout['config']);ready=read(c['preparation']['path']);base=Path(ready['base'])
    need(c.get('mode','baseline')==layout['mode'],'JOB_CONFIG_MODE_MISMATCH')
    need(read(training/'status.json')['status']=='COMPLETED','TRAINING_NOT_COMPLETE')
    gpu=read(root/'job_spec.json')['gpu_uuid']
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,nnUNet_preprocessed=str(base.parent),nnUNet_results=str(training/'results'),nnUNet_compile='false',nnUNet_n_proc_DA='0',
                      LESIONGEN512_POOL=str(Path(ready['pool_manifest']).parent),LESIONGEN512_POOL_SHA256=ready['pool_manifest_sha256'],
                      DATASET509_QUALITY_CONFIG=str(training/'config.json'),nnUNet_def_n_proc='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    repo=Path(c['lesiongen_repo']);sys.path[:0]=[str(repo/'src'),str(repo/'trainers')]
    import torch
    torch.set_num_threads(1)
    from trainer import trainer_class
    from api_contract import verify
    verify()
    trainer=trainer_class(layout['mode'])(read(ready['plans']),'2d',0,read(base/'dataset.json'),torch.device('cuda:0'))
    best=Path(trainer.output_folder)/'checkpoint_best.pth'
    need(best.is_file(),'BEST_CHECKPOINT_MISSING')
    trainer.load_checkpoint(str(best))
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    trainer.dataset_class=infer_dataset_class(trainer.preprocessed_dataset_folder)
    trainer.perform_actual_validation(save_probabilities=False)
    predictions=Path(trainer.output_folder)/'validation';split=read(c['split']['path'])[0]
    need({x.name[:-7] for x in predictions.glob('*.nii.gz')}==set(split['val']) and len(split['val'])==28,'VALIDATION_COHORT_MISMATCH')
    metrics=layout['metrics'];need(not metrics.exists(),'METRICS_ALREADY_EXIST')
    gt=Path(read(root/'job_spec.json')['source_dataset509'])/'labelsTr'
    subprocess.run([sys.executable,str(repo/'scripts/evaluate_segmentation_metrics.py'),'--pred-dir',str(predictions),'--gt-dir',str(gt),'--output-dir',str(metrics),'--f2-mode','paper'],check=True)
    import csv
    with (metrics/'case_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    supplement(predictions,gt,rows,metrics/'supplemental_summary.json')
    write_new(root/'evaluation.status.json',{'status':'COMPLETED','validation_volumes':28,'evaluation_space':'ORIGINAL_FULL_VOLUME','checkpoint_sha256':sha(best),'metrics':str(metrics),'f2_mode':'paper_equivalent_F0.5','fold':0})


if __name__=='__main__':main()