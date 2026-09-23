"""Oracle train-box coverage only: expanded aspect ratios are not an AP result."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;rows=json.loads((a.run/'result.json').read_text())['rows'];out=[]
    for row in rows:
        with Image.open(root/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'/f'{row["image_id"]}.jpg') as im:ow,oh=im.size
        scale=480/max(ow,oh);w=int(ow*scale+.5);h=int(oh*scale+.5);boxes=[]
        for side in (64,96,128,160,192,224,256):
            for ratio in (.5,2.):
                rw=round(side*math.sqrt(ratio));rh=round(side/math.sqrt(ratio))
                boxes.extend([x,y,x+rw-1,y+rh-1] for y in range(0,h-rh+1,32) for x in range(0,w-rw+1,32))
        xy=np.array(boxes);b=np.array(row['object_box'])
        inter=np.maximum(0,np.minimum(xy[:,2:],b[2:])-np.maximum(xy[:,:2],b[:2])+1).prod(1)
        area=(xy[:,2:]-xy[:,:2]+1).prod(1);ba=(b[2:]-b[:2]+1).prod();iou=inter/(area+ba-inter)
        out.append(dict(cls=row['cls'],id=row['image_id'],square_iou=row['roi_iou'],expanded_iou=max(row['roi_iou'],float(iou.max())),extra_regions=len(boxes)))
    summary={c:dict(square_median=float(np.median([x['square_iou'] for x in out if x['cls']==c])),expanded_median=float(np.median([x['expanded_iou'] for x in out if x['cls']==c])),mean_extra_regions=float(np.mean([x['extra_regions'] for x in out if x['cls']==c]))) for c in ('bottle','pottedplant','sofa')}
    (a.run/'geometry_check.json').write_text(json.dumps(dict(rows=out,summary=summary,limitation='Oracle best IoU using train boxes; extra region budget, no inference or AP gain measured.'),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
