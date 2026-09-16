"""Read detected QR contents and optionally save JSON and annotated PNG."""
import argparse
import json
from pathlib import Path
import sys
from reader import CodeReader, DEFAULT_WEIGHTS, load_image, annotate

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=1024)
    p.add_argument('--mode',choices=['yolo','decoder'],default='yolo')
    p.add_argument('--output',type=Path)
    p.add_argument('--annotated',type=Path)
    a=p.parse_args()
    image=load_image(a.image)
    result=CodeReader(a.weights).read(image,a.confidence,a.imgsz,a.mode)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if a.output:
        a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(text,encoding='utf-8')
    if a.annotated:
        a.annotated.parent.mkdir(exist_ok=True,parents=True);annotate(image,result).save(a.annotated)
