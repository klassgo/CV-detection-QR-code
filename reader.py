"""YOLO localizes; ZXing decodes. All public image arrays are RGB uint8."""
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import zxingcpp

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'best.pt'


def load_image(source):
    if isinstance(source, bytes):
        source = BytesIO(source)
    with Image.open(source) as im:
        if im.width * im.height > 25_000_000:
            raise ValueError('Изображение слишком большое: максимум 25 мегапикселей.')
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def decode_crop(image, box, padding=.15, qr_only=True):
    h,w=image.shape[:2]
    x1,y1,x2,y2=map(float,box)
    margin=max(8,int(max(x2-x1,y2-y1)*padding))
    left,top=max(0,int(np.floor(x1))-margin),max(0,int(np.floor(y1))-margin)
    right,bottom=min(w,int(np.ceil(x2))+margin),min(h,int(np.ceil(y2))+margin)
    if right<=left or bottom<=top: return []
    crop=image[top:bottom,left:right]
    kwargs={'formats':zxingcpp.BarcodeFormat.QRCode} if qr_only else {}
    found=zxingcpp.read_barcodes(crop,**kwargs)
    scale=1
    if not found and max(crop.shape[:2])<=1500:
        scale=2
        enlarged=np.asarray(Image.fromarray(crop).resize((crop.shape[1]*2,crop.shape[0]*2),Image.Resampling.LANCZOS))
        found=zxingcpp.read_barcodes(enlarged,**kwargs)
    results=[]
    for code in found:
        p=code.position
        points=[[round(v.x/scale+left,2),round(v.y/scale+top,2)]
                for v in [p.top_left,p.top_right,p.bottom_right,p.bottom_left]]
        xs,ys=zip(*points)
        # Do not attribute a neighboring code, caught by crop padding, to this detection.
        if not (x1<=(min(xs)+max(xs))/2<=x2 and y1<=(min(ys)+max(ys))/2<=y2): continue
        results.append({'text':code.text,'bytes_hex':bytes(code.bytes).hex(), 'format':str(code.format),
                        'bbox':[min(xs),min(ys),max(xs),max(ys)],'polygon':points})
    return results


class CodeReader:
    def __init__(self,weights=DEFAULT_WEIGHTS):
        weights=Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'Не найдены веса: {weights}')
        os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime/ultralytics'))
        from ultralytics import YOLO
        self.model=YOLO(str(weights),task='detect')
        self.weights=weights.name
        self.lock=Lock()

    def read(self,image,confidence=.25,imgsz=1024,mode='yolo',qr_only=True):
        if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected RGB uint8 image')
        if mode not in {'yolo','decoder'}: raise ValueError(mode)
        started=perf_counter()
        detector_ms=0.0
        if mode=='yolo':
            with self.lock:
                prediction=self.model.predict(source=Image.fromarray(image),conf=confidence,imgsz=imgsz,
                                              device='cpu',verbose=False,save=False)[0]
            detector_ms=(perf_counter()-started)*1000
            boxes=prediction.boxes
            candidates=[{'bbox':list(map(float,xy)), 'confidence':float(score),
                         'class':prediction.names[int(cls)]}
                        for xy,score,cls in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist())]
        else:
            h,w=image.shape[:2]
            candidates=[{'bbox':[0,0,w,h],'confidence':None,'class':'decoder'}]
        detections=[]; codes=[]
        for candidate in candidates:
            decoded=decode_crop(image,candidate['bbox'],qr_only=qr_only)
            ids=[]
            for item in decoded:
                existing=next((c for c in codes if c['bytes_hex']==item['bytes_hex'] and c['format']==item['format']
                               and overlap(c['bbox'],item['bbox'])>.3),None)
                if existing is None:
                    item['id']=len(codes)+1; codes.append(item); existing=item
                ids.append(existing['id'])
            if mode=='yolo':
                detections.append({'id':len(detections)+1,**candidate,'status':'decoded' if ids else 'unreadable','code_ids':ids})
        if mode=='decoder':
            detections=[{'id':i+1,'bbox':c['bbox'],'confidence':None,'class':c['format'],'status':'decoded','code_ids':[c['id']]}
                        for i,c in enumerate(codes)]
        elapsed=(perf_counter()-started)*1000
        return {'mode':mode,'weights':self.weights if mode=='yolo' else None,
                'model_classes':self.model.names if mode=='yolo' else None,
                'image_size':{'width':image.shape[1],'height':image.shape[0]},
                'settings':{'confidence':confidence,'imgsz':imgsz,'device':'cpu','qr_only':qr_only},
                'detections':detections,'codes':codes,
                'timing_ms':{'total':round(elapsed,1),'detector':round(detector_ms,1),'decoder_and_postprocess':round(elapsed-detector_ms,1)}}


def annotate(image,report):
    result=Image.fromarray(image).copy(); draw=ImageDraw.Draw(result)
    width=max(2,round(max(result.size)/400))
    for d in report['detections']:
        color='#137C66' if d['status']=='decoded' else '#D97706'
        box=d['bbox']; draw.rectangle(box,outline=color,width=width)
        label=f"#{d['id']} " + ('READ' if d['status']=='decoded' else 'UNREAD')
        x,y=max(0,box[0]),max(0,box[1]-18)
        draw.rectangle([x,y,x+100,y+18],fill=color)
        draw.text((x+4,y+2),label,fill='white')
    return result
