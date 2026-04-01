"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, rt="data/oxford-pets", s="trainval"):
        super().__init__()
        self.rt = rt
        self.s = s
        self.sz = 224
        
        self.img_d = os.path.join(rt, "images")
        self.ann_d = os.path.join(rt, "annotations", "xmls")
        self.tri_d = os.path.join(rt, "annotations", "trimaps")
        
        txt = os.path.join(rt, "annotations", f"{s}.txt")
        self.dt = []
        
        if not os.path.exists(txt):
            raise FileNotFoundError(
                f"Dataset not found at {txt}. "
                "Please download the Oxford-IIIT Pet dataset (images and annotations), "
                "extract them, and place them inside the 'data/oxford-pets' folder."
            )
            
        with open(txt, 'r') as f:
            for l in f:
                    if l.startswith('#'): continue
                    p = l.strip().split()
                    if len(p) >= 2:
                        self.dt.append({
                            'nm': p[0],
                            'cls': int(p[1]) - 1 
                        })
        
        self.trn = T.Compose([
            T.Resize((self.sz, self.sz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, i):
        itm = self.dt[i]
        nm = itm['nm']
        cls = itm['cls']
        
        im_p = os.path.join(self.img_d, f"{nm}.jpg")
        im = Image.open(im_p).convert("RGB")
        ow, oh = im.size
        t_im = self.trn(im)
        
        tm_p = os.path.join(self.tri_d, f"{nm}.png")
        if os.path.exists(tm_p):
            tm = Image.open(tm_p)
            tm = tm.resize((self.sz, self.sz), Image.NEAREST)
            tm_t = torch.as_tensor(list(tm.getdata()), dtype=torch.long).view(self.sz, self.sz)
            tm_t = tm_t - 1 
            tm_t[tm_t < 0] = 2 
        else:
            tm_t = torch.zeros((self.sz, self.sz), dtype=torch.long)
            
        xl_p = os.path.join(self.ann_d, f"{nm}.xml")
        bx = torch.zeros(4)
        if os.path.exists(xl_p):
            tr = ET.parse(xl_p).getroot()
            b_ob = tr.find('object').find('bndbox')
            xmin = float(b_ob.find('xmin').text)
            ymin = float(b_ob.find('ymin').text)
            xmax = float(b_ob.find('xmax').text)
            ymax = float(b_ob.find('ymax').text)
            
            nx1 = xmin * (self.sz / ow)
            ny1 = ymin * (self.sz / oh)
            nx2 = xmax * (self.sz / ow)
            ny2 = ymax * (self.sz / oh)
            
            cx = (nx1 + nx2) / 2.0
            cy = (ny1 + ny2) / 2.0
            w = nx2 - nx1
            h = ny2 - ny1
            
            bx = torch.tensor([cx, cy, w, h])
            
        return t_im, cls, bx, tm_t