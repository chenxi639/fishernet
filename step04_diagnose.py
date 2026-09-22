from pathlib import Path
import sys,json,os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root=Path(__file__).resolve().parent
voc=root/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
source=root/'runs/step04_continue_20260919_154811'
out=root/'report/assets/weak_diagnosis'
out.mkdir(parents=True,exist_ok=True)
with np.load(source/'best_predictions.npz') as p:
 scores,labels,ids=p['scores'],p['labels'],p['image_ids']
classes=list(json.loads((source/'result.json').read_text())['best_metrics']['ap'])
result={}
for c in ('bottle','sofa','pottedplant'):
 j=classes.index(c); pos=np.flatnonzero(labels[:,j]==1); neg=np.flatnonzero(labels[:,j]==-1)
 boxes={}; stats=[]
 for i in pos:
  tree=ET.parse(voc/'Annotations'/f'{ids[i]}.xml').getroot()
  w,h=float(tree.findtext('size/width')),float(tree.findtext('size/height'))
  bb=[]
  for o in tree.findall('object'):
   if o.findtext('name')==c and o.findtext('difficult','0')=='0':
    b=[float(o.findtext('bndbox/'+key)) for key in ('xmin','ymin','xmax','ymax')]
    bb.append(b)
  boxes[int(i)]=bb
  largest=max(bb,key=lambda b:(b[2]-b[0]+1)*(b[3]-b[1]+1))
  area=(largest[2]-largest[0]+1)*(largest[3]-largest[1]+1)/(w*h)
  width224=(largest[2]-largest[0]+1)/w*224
  stats.append((int(i),area,width224))
 stats=np.array(stats)
 fn=scores[pos,j]<0
 result[c]={'positive_images':len(pos),'negative_images':len(neg),'false_negative_at_point5':int(fn.sum()),
  'false_positive_at_point5':int((scores[neg,j]>=0).sum()),
  'median_largest_area_fraction':float(np.median(stats[:,1])),
  'median_largest_width_at224':float(np.median(stats[:,2])),
  'fn_median_largest_area_fraction':float(np.median(stats[fn,1])),
  'tp_median_largest_area_fraction':float(np.median(stats[~fn,1]))}
 # Explicitly mark ranking extremes, not cherry-picked random examples.
 worst_pos=pos[np.argsort(scores[pos,j])[:4]]
 worst_neg=neg[np.argsort(-scores[neg,j])[:4]]
 chosen=np.r_[worst_pos,worst_neg]
 result[c]['examples']=[{'id':str(ids[i]),'raw_label':int(labels[i,j]),'logit':float(scores[i,j])} for i in chosen]
 fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
 for ax,i in zip(axes.ravel(),chosen):
  with Image.open(voc/'JPEGImages'/f'{ids[i]}.jpg') as im: ax.imshow(im.convert('RGB'))
  for b in boxes.get(int(i),[]): ax.add_patch(Rectangle((b[0]-1,b[1]-1),b[2]-b[0]+1,b[3]-b[1]+1,fill=False,color='red',lw=1.5))
  ax.set_title(f'{ids[i]} | y={labels[i,j]} | logit={scores[i,j]:.2f}',fontsize=9); ax.axis('off')
 fig.suptitle(c+': lowest-scored positives (top) / highest-scored negatives (bottom)')
 fig.savefig(out/f'{c}_errors.png',dpi=140); plt.close(fig)
(out/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in result.items()},indent=2))
