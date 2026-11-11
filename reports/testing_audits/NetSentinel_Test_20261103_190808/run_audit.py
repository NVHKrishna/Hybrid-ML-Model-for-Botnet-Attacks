import json, os, sys, traceback, platform, importlib.metadata as im
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs'/'NetSentinel_20261103_113830'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
checks=[]
def record(name,status,detail): checks.append({'test':name,'status':status,'detail':str(detail)})
def safe(name, fn):
 try: record(name,'PASS',fn())
 except Exception as e: record(name,'FAIL',f'{type(e).__name__}: {e}')

def main():
 import tensorflow as tf
 from src.data_preprocessing import load_dataset, build_paper_splits, PaperPreprocessor
 from src.evaluate import compute_metrics
 from src.models import predict_proba
 cfg=json.loads('{}')
 import yaml
 cfg=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf-8'))
 metadata=json.loads((RUN/'reports/run_metadata.json').read_text(encoding='utf-8'))
 models={x:RUN/'models'/f'{x}_best.keras' for x in ['ann','cnn','lstm','rnn']}
 models['aclr']=RUN/'models/aclr_meta.keras'
 prep_path=RUN/'models/preprocessor.joblib'
 record('Python/runtime versions','PASS',{'python':platform.python_version(),'tensorflow':tf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':im.version('scikit-learn')})
 record('Required package versions','PASS',{p:im.version(p) for p in ['tensorflow','scikit-learn','numpy','pandas','matplotlib','seaborn','joblib','scipy','PyYAML']})
 record('Configured dataset/checkpoint/preprocessor paths','PASS',{'dataset':cfg['data']['dataset_file'],'dataset_exists':(ROOT/cfg['data']['dataset_file']).is_file(),'checkpoints':{k:v.is_file() for k,v in models.items()},'preprocessor_exists':prep_path.is_file(),'old_legacy_checkpoints_present':{p.name:(ROOT/'models'/p).exists() for p in [Path('ann_final.keras'),Path('cnn_final.keras'),Path('lstm_final.keras'),Path('rnn_final.keras')]}})
 df=load_dataset(str(ROOT/cfg['data']['dataset_file']))
 Xraw=df.drop(columns=['label','attack_cat','id'],errors='ignore'); y=df.label.to_numpy(dtype=np.float32)
 duplicates=int(df.duplicated().sum()); nulls=int(df.isna().sum().sum()); numeric=Xraw.select_dtypes(include=[np.number]).to_numpy(); inf=int(np.isinf(numeric).sum())
 record('Dataset dimensions/distribution/quality','PASS',{'rows':len(df),'columns':len(df.columns),'features':Xraw.shape[1],'labels':{str(int(k)):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'missing_cells':nulls,'duplicate_rows':duplicates,'numeric_infinite_values':inf,'feature_columns_include_target':bool({'label','attack_cat'} & set(Xraw.columns))})
 splits=build_paper_splits(str(ROOT/cfg['data']['dataset_file']),cfg['data']['test_split'],cfg['data']['val_split'],cfg['training']['random_seed'])
 disjoint=(not(set(splits['train_indices'])&set(splits['val_indices'])) and not(set(splits['train_indices'])&set(splits['test_indices'])) and not(set(splits['val_indices'])&set(splits['test_indices'])))
 record('Partitioning and train-only preprocessing fit','PASS' if disjoint and splits['preprocessor'].scaler_.n_samples_seen_==len(splits['y_train']) else 'FAIL',{'sizes':{k:len(splits[k]) for k in ['y_train','y_val','y_test']},'pairwise_disjoint':disjoint,'scaler_n_samples_seen':int(splits['preprocessor'].scaler_.n_samples_seen_),'train_rows':len(splits['y_train']),'features':splits['feature_names'],'no_target_columns':not(bool({'label','attack_cat','id'}&set(splits['feature_names'])))})
 prep=PaperPreprocessor.load(str(prep_path))
 record('Saved preprocessor matches recomputed training fit','PASS' if prep.feature_columns_==splits['feature_names'] and prep.scaler_.n_samples_seen_==len(splits['y_train']) else 'FAIL',{'saved_features_match':prep.feature_columns_==splits['feature_names'],'saved_scaler_samples':int(prep.scaler_.n_samples_seen_)})
 # unseen and order probes
 raw_sample=Xraw.iloc[:4].copy(); base=prep.transform(raw_sample); shuffled=prep.transform(raw_sample[prep.feature_columns_[::-1]])
 unknown=raw_sample.iloc[:1].copy(); unknown['service']='__audit_unseen_category__'; unknown_z=prep.transform(unknown)
 record('Categorical unknown and feature-order behavior','PASS' if np.allclose(base,shuffled) and np.isfinite(unknown_z).all() else 'FAIL',{'reordered_features_identical':bool(np.allclose(base,shuffled)),'unknown_category_finite':bool(np.isfinite(unknown_z).all())})
 # heldout stored targets/preds
 y_saved=np.load(RUN/'outputs/y_test.npy',allow_pickle=False)
 record('Saved labels align with deterministic held-out split','PASS' if np.array_equal(y_saved,splits['y_test']) else 'FAIL',{'shape':y_saved.shape,'equal_to_recomputed_test_y':bool(np.array_equal(y_saved,splits['y_test']))})
 saved_metrics=json.loads((RUN/'reports/metrics_test.json').read_text(encoding='utf-8'))
 all_metrics={}; cmats={}; inference={}
 loaded={}
 for name,path in models.items():
  try:
   m=tf.keras.models.load_model(path); loaded[name]=m
   expected=(None,42) if name=='ann' else ((None,4) if name=='aclr' else (None,42,1))
   record(f'{name.upper()} checkpoint load and input/output architecture','PASS' if tuple(m.input_shape)==expected and tuple(m.output_shape)==(None,1) else 'FAIL',{'input_shape':m.input_shape,'output_shape':m.output_shape,'file_bytes':path.stat().st_size})
  except Exception as e: record(f'{name.upper()} checkpoint load and input/output architecture','FAIL',f'{type(e).__name__}: {e}')
 seq={'cnn','lstm','rnn'}
 for name in ['ann','cnn','lstm','rnn']:
  if name not in loaded: continue
  model=loaded[name]; isseq=name in seq
  ptest=predict_proba(model,splits['X_test'],reshape_seq=isseq)
  pval=predict_proba(model,splits['X_val'],reshape_seq=isseq)
  saved=np.load(RUN/f'outputs/prob_{name}_test.npy',allow_pickle=False)
  valid=ptest.shape==y_saved.shape and pval.shape==(len(splits['y_val']),) and np.isfinite(ptest).all() and np.isfinite(pval).all() and (ptest>=0).all() and (ptest<=1).all()
  record(f'{name.upper()} validation/test inference outputs','PASS' if valid else 'FAIL',{'val_shape':pval.shape,'test_shape':ptest.shape,'finite':bool(np.isfinite(ptest).all()),'probability_range':[float(ptest.min()),float(ptest.max())],'saved_array_max_abs_diff':float(np.max(np.abs(ptest-saved)))})
  repeated=predict_proba(model,splits['X_test'][:5],reshape_seq=isseq)
  repeated2=predict_proba(model,splits['X_test'][:5],reshape_seq=isseq)
  record(f'{name.upper()} repeated inference consistency','PASS' if np.allclose(repeated,repeated2,rtol=0,atol=1e-7) else 'FAIL',{'max_abs_diff':float(np.max(np.abs(repeated-repeated2)))})
  # single and batch input, normal/attack examples
  one=predict_proba(model,splits['X_test'][:1],reshape_seq=isseq); batch=predict_proba(model,splits['X_test'][:8],reshape_seq=isseq)
  record(f'{name.upper()} single and multiple row inference','PASS' if one.shape==(1,) and batch.shape==(8,) else 'FAIL',{'single':one.tolist(),'batch_shape':batch.shape})
  all_metrics[name.upper()]={
   'accuracy':accuracy_score(y_saved,ptest>=0.5),'precision':precision_score(y_saved,ptest>=0.5,zero_division=0),'recall':recall_score(y_saved,ptest>=0.5,zero_division=0),'f1':f1_score(y_saved,ptest>=0.5,zero_division=0),'roc_auc':roc_auc_score(y_saved,ptest),'pr_auc':average_precision_score(y_saved,ptest)}
  cmats[name.upper()]=confusion_matrix(y_saved,ptest>=0.5).tolist()
  inference[name]=ptest
 # ACLR exactly four base columns, full flow using loaded run-specific base checkpoints
 if all(x in loaded for x in ['ann','cnn','lstm','rnn','aclr']):
  base_test=np.column_stack([inference[x] for x in ['ann','cnn','lstm','rnn']]).astype(np.float32)
  base_val=np.column_stack([predict_proba(loaded[x],splits['X_val'],reshape_seq=x in seq) for x in ['ann','cnn','lstm','rnn']]).astype(np.float32)
  aclr=loaded['aclr']; meta_test=aclr.predict(base_test,verbose=0).ravel(); meta_val=aclr.predict(base_val,verbose=0).ravel()
  saved_aclr=np.load(RUN/'outputs/prob_aclr_test.npy',allow_pickle=False)
  ok=base_test.shape==(len(y_saved),4) and np.isfinite(meta_test).all() and (meta_test>=0).all() and (meta_test<=1).all() and np.allclose(meta_test,saved_aclr,atol=1e-6)
  record('ACLR four-feature complete stack validation/test inference','PASS' if ok else 'FAIL',{'meta_input_shape':aclr.input_shape,'base_test_shape':base_test.shape,'val_output_shape':meta_val.shape,'test_output_shape':meta_test.shape,'saved_array_max_abs_diff':float(np.max(np.abs(meta_test-saved_aclr)))})
  all_metrics['ACLR']={'accuracy':accuracy_score(y_saved,meta_test>=0.5),'precision':precision_score(y_saved,meta_test>=0.5,zero_division=0),'recall':recall_score(y_saved,meta_test>=0.5,zero_division=0),'f1':f1_score(y_saved,meta_test>=0.5,zero_division=0),'roc_auc':roc_auc_score(y_saved,meta_test),'pr_auc':average_precision_score(y_saved,meta_test)}
  cmats['ACLR']=confusion_matrix(y_saved,meta_test>=0.5).tolist()
  record('ACLR single/multiple inference','PASS' if aclr.predict(base_test[:1],verbose=0).shape==(1,1) and aclr.predict(base_test[:8],verbose=0).shape==(8,1) else 'FAIL',{'single_shape':aclr.predict(base_test[:1],verbose=0).shape,'batch_shape':aclr.predict(base_test[:8],verbose=0).shape})
 # invalid input probes are validation-only; no fitting or writes
 for modelname in ['ann','cnn','lstm','rnn']:
  if modelname not in loaded: continue
  model=loaded[modelname]; wrong=np.zeros((2,41),np.float32); wrong=np.expand_dims(wrong,-1) if modelname in seq else wrong
  try: model.predict(wrong,verbose=0); rejected=False
  except Exception: rejected=True
  record(f'{modelname.upper()} invalid feature dimension rejection','PASS' if rejected else 'FAIL','invalid input was rejected' if rejected else 'invalid input unexpectedly accepted')
 for label,fn in [('missing feature column',lambda:prep.transform(raw_sample.drop(columns=[prep.feature_columns_[0]]))),('missing numeric value',lambda:prep.transform(raw_sample.assign(**{prep.feature_columns_[0]:np.nan}))),('unexpected string numeric type',lambda:prep.transform(raw_sample.assign(**{next(c for c in prep.feature_columns_ if c not in prep.categorical_cols):'not_numeric'})))]:
  try: fn(); rejected=False
  except Exception as e: rejected=True; reason=f'{type(e).__name__}: {e}'
  record(f'Input validation: {label}','PASS' if rejected else 'FAIL',reason if rejected else 'input unexpectedly accepted')
 # nonexistent model path check, no creation
 try: tf.keras.models.load_model(OUT/'deliberately_missing_checkpoint.keras'); rejected=False
 except Exception as e: rejected=True; reason=f'{type(e).__name__}: missing checkpoint rejected'
 record('Missing checkpoint failure handling','PASS' if rejected else 'FAIL',reason if rejected else 'missing checkpoint unexpectedly loaded')
 # independently compute metrics and reconcile saved metrics
 metric_rows=[]
 for name,metrics in all_metrics.items():
  expected=saved_metrics[name]
  diffs={k:abs(float(metrics[k])-float(expected[k])) for k in metrics}
  metric_rows.append({'model':name,'recomputed':{k:float(v) for k,v in metrics.items()},'saved':expected,'absolute_differences':diffs,'confusion_matrix_tn_fp_fn_tp':cmats[name]})
  record(f'{name} independent saved test metric reconciliation','PASS' if max(diffs.values())<=1.1e-6 else 'FAIL',{'max_abs_diff':max(diffs.values()),'confusion_matrix':cmats[name]})
 (OUT/'independent_metrics.json').write_text(json.dumps(metric_rows,indent=2),encoding='utf-8')
 # static execution evidence for OOF plus metadata, no claim external kfold
 log=(RUN/'logs/train.log').read_text(encoding='utf-8',errors='replace')
 oof_count=log.count('Fold ')
 record('Saved training log OOF fold evidence','PASS' if oof_count>=5 and 'Execution complete.' in log else 'BLOCKED',{'fold_marker_count':oof_count,'execution_complete_marker':'Execution complete.' in log,'metadata_oof_folds':metadata.get('oof_folds'),'external_full_data_kfold':False})
 # report test suite has training and project mutation side-effects; deliberately not run.
 record('Existing tests/test_pipeline.py execution','NOT TESTED','Not run: the script fits base models and ACLR, writes under models/, and deletes files; explicit no-retraining constraint takes precedence.')
 (OUT/'test_results.json').write_text(json.dumps({'audit_dir':str(OUT),'run_dir':str(RUN),'checks':checks,'metric_rows':metric_rows,'oof_log_fold_marker_count':oof_count},indent=2),encoding='utf-8')
 print(json.dumps({'audit_dir':str(OUT),'check_counts':{s:sum(c['status']==s for c in checks) for s in ['PASS','FAIL','BLOCKED','NOT TESTED']},'checks':checks,'metrics':metric_rows},indent=2))
if __name__=='__main__': main()

