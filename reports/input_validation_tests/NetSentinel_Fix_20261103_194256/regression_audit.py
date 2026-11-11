import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs'/'NetSentinel_20261103_113830'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.models import build_ann,build_cnn,build_lstm,build_rnn,predict_proba
from src.stacking import build_aclr_meta
from src.data_preprocessing import build_paper_splits,PaperPreprocessor,load_dataset
import tensorflow as tf
checks=[]
def add(name,passed,detail=''):
 checks.append({'name':name,'status':'PASS' if passed else 'FAIL','detail':str(detail)})
def must_raise(name,fn):
 try: fn(); add(name,False,'input unexpectedly accepted')
 except ValueError as e: add(name,True,f'{type(e).__name__}: {e}')
 except Exception as e: add(name,False,f'expected informative ValueError; got {type(e).__name__}: {e}')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 builders={'ANN':(build_ann,False),'CNN':(build_cnn,True),'LSTM':(build_lstm,True),'RNN':(build_rnn,True)}
 build_specs=[]
 # Fresh builders: valid inputs and malformed tensors.
 x=np.zeros((3,42),np.float32)
 for name,(builder,seq) in builders.items():
  m=builder(42); xin=x[...,None] if seq else x
  y=m.predict(xin,verbose=0)
  add(f'builder {name} valid 42-feature inference',y.shape==(3,1) and np.isfinite(y).all() and (y>=0).all() and (y<=1).all(),{'shape':y.shape,'range':[float(y.min()),float(y.max())]})
  for label,bad in [('41 features',np.zeros((2,41,1),np.float32) if seq else np.zeros((2,41),np.float32)),('43 features',np.zeros((2,43,1),np.float32) if seq else np.zeros((2,43),np.float32)),('rank 1',np.zeros((42,),np.float32)),('wrong rank/channel',np.zeros((2,42,2),np.float32) if seq else np.zeros((2,42,1),np.float32))]:
   must_raise(f'builder {name} rejects {label}',lambda m=m,bad=bad:m.predict(bad,verbose=0))
 # Load saved checkpoints: no model files are written; predictions via public helper
 # install runtime InputSpec, then probe direct calls too.
 paths={n:RUN/'models'/f'{n.lower()}_best.keras' for n in builders}
 paths['ACLR']=RUN/'models'/'aclr_meta.keras'
 before={n:sha(p) for n,p in paths.items()}
 loaded={n:tf.keras.models.load_model(p) for n,p in paths.items()}
 for name,(builder,seq) in builders.items():
  m=loaded[name]; Xin=x[...,None] if seq else x
  p=predict_proba(m,x,reshape_seq=seq)
  add(f'checkpoint {name} valid 42-feature inference',p.shape==(3,) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),{'shape':p.shape,'range':[float(p.min()),float(p.max())]})
  for label,bad in [('41 features',np.zeros((2,41,1),np.float32) if seq else np.zeros((2,41),np.float32)),('43 features',np.zeros((2,43,1),np.float32) if seq else np.zeros((2,43),np.float32)),('rank 1',np.zeros((42,),np.float32)),('wrong rank/channel',np.zeros((2,42,2),np.float32) if seq else np.zeros((2,42,1),np.float32))]:
   must_raise(f'checkpoint {name} direct rejects {label}',lambda m=m,bad=bad:m.predict(bad,verbose=0))
 # ACLR is a 4-probability-feature model, not a raw 42-feature model.
 meta=loaded['ACLR']; z=np.zeros((3,4),np.float32); q=meta.predict(z,verbose=0)
 add('ACLR valid four-base-probability input',q.shape==(3,1) and np.isfinite(q).all() and (q>=0).all() and (q<=1).all(),{'shape':q.shape,'range':[float(q.min()),float(q.max())]})
 for label,bad in [('41 columns',np.zeros((2,41),np.float32)),('5 columns',np.zeros((2,5),np.float32)),('rank 1',np.zeros((4,),np.float32)),('wrong rank',np.zeros((2,4,1),np.float32))]:
  must_raise(f'ACLR rejects {label}',lambda bad=bad:meta.predict(bad,verbose=0))
 # public helper validates all wrong shapes with informative ValueError
 for name,(builder,seq) in builders.items():
  m=loaded[name]
  for label,bad in [('41 features',np.zeros((2,41),np.float32)),('43 features',np.zeros((2,43),np.float32)),('rank 1',np.zeros((42,),np.float32)),('rank 3 input matrix',np.zeros((2,42,1),np.float32))]:
   must_raise(f'predict_proba {name} rejects {label}',lambda m=m,bad=bad,seq=seq:predict_proba(m,bad,reshape_seq=seq))
 # Full saved preprocessing to ACLR pipeline; compare every test result to prior arrays.
 splits=build_paper_splits(str(ROOT/'dataset'/'UNSW_NB15_training-set.csv'),.30,.15,42)
 pre=PaperPreprocessor.load(str(RUN/'models'/'preprocessor.joblib'))
 df=load_dataset(str(ROOT/'dataset'/'UNSW_NB15_training-set.csv'))
 raw=df.drop(columns=['label','attack_cat','id'])
 Xtest=pre.transform(raw.iloc[splits['test_indices']])
 preds={}
 for name,(builder,seq) in builders.items():
  preds[name.lower()]=predict_proba(loaded[name],Xtest,reshape_seq=seq)
  saved=np.load(RUN/'outputs'/f'prob_{name.lower()}_test.npy',allow_pickle=False)
  add(f'end-to-end {name} predictions unchanged',preds[name.lower()].shape==saved.shape and np.allclose(preds[name.lower()],saved,rtol=0,atol=1e-5),{'max_abs_diff':float(np.max(np.abs(preds[name.lower()]-saved)))})
 stack_features=np.column_stack([preds[n] for n in ['ann','cnn','lstm','rnn']]).astype(np.float32)
 aclr=loaded['ACLR'].predict(stack_features,verbose=0).ravel()
 saved=np.load(RUN/'outputs'/'prob_aclr_test.npy',allow_pickle=False)
 add('end-to-end preprocessing → four base models → ACLR unchanged',aclr.shape==saved.shape and np.allclose(aclr,saved,rtol=0,atol=1e-5),{'stack_input_shape':stack_features.shape,'output_shape':aclr.shape,'max_abs_diff':float(np.max(np.abs(aclr-saved)))})
 after={n:sha(p) for n,p in paths.items()}
 add('Saved checkpoints remain byte-for-byte unchanged',before==after,{'before_equals_after':before==after})
 (OUT/'results.json').write_text(json.dumps({'checks':checks,'counts':{s:sum(c['status']==s for c in checks) for s in ['PASS','FAIL']},'checkpoint_sha256':before},indent=2),encoding='utf-8')
 print(json.dumps({'counts':{s:sum(c['status']==s for c in checks) for s in ['PASS','FAIL']},'checks':checks},indent=2))
if __name__=='__main__': main()

