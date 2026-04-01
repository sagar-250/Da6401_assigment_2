"""Training entrypoint
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

def trn_net(net, t_loader, tsk, ep=20):
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    c_loss = nn.CrossEntropyLoss()
    m_loss = nn.MSELoss()
    i_loss = IoULoss(reduction='mean')
    
    bst_loss = float('inf')
    
    for e in range(ep):
        net.train()
        tl = 0.0
        
        for k, (im, cls, bx, mask) in enumerate(t_loader):
            im = im.cuda()
            opt.zero_grad()
            
            if tsk == 'cls':
                cls = cls.cuda()
                out = net(im)
                l = c_loss(out, cls)
            elif tsk == 'loc':
                bx = bx.cuda()
                out = net(im)
                l = m_loss(out, bx) + i_loss(out, bx)
            elif tsk == 'seg':
                mask = mask.cuda()
                out = net(im)
                l = c_loss(out, mask)
                
            l.backward()
            opt.step()
            tl += l.item()
            
            if k % 10 == 0:
                print(f"[{tsk}] Ep {e} B {k} Ls: {l.item():.4f}")
                wandb.log({f"{tsk}_loss": l.item()})
                
        al = tl / len(t_loader)
        print(f"Ep {e} avg: {al:.4f}")
        
        # simple save logic (save only best)
        if al < bst_loss:
            bst_loss = al
            print(f"Saving best model for {tsk} with loss {bst_loss:.4f}...")
            if tsk == 'cls': nm = "classifier.pth"
            elif tsk == 'loc': nm = "localizer.pth"
            else: nm = "unet.pth"
                
            sv = {
                "state_dict": net.state_dict(),
                "epoch": e,
                "best_metric": al
            }
            torch.save(sv, f"checkpoints/{nm}")

if __name__ == "__main__":
    wandb.init(project="da6401_assignment_2")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    bdir = os.path.dirname(os.path.abspath(__file__))
    rp = os.path.join(bdir, 'data', 'oxford-pets')
    
    ds = OxfordIIITPetDataset(rt=rp)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    
    c_net = VGG11Classifier().cuda()
    print("Train Class...")
    trn_net(c_net, dl, 'cls', ep=20)
    
    # Task 2: Isolate convolutional backbone from Task 1
    # We choose to FINE-TUNE the pretrained weights rather than freezing them. 
    # Justification: While the classifier learns good semantic features, localization 
    # requires explicit spatial awareness. Fine-tuning allows the encoder to adapt 
    # its receptive fields specifically for bounding box coordinate regression.
    l_net = VGG11Localizer().cuda()
    l_net.enc.load_state_dict(c_net.enc.state_dict())
    print("\nTrain Loc (Fine-tuning Task 1 Encoder)...")
    trn_net(l_net, dl, 'loc', ep=20)
    
    # Task 3: Use Task 1 encoder for U-Net contracting path
    s_net = VGG11UNet().cuda()
    s_net.enc.load_state_dict(c_net.enc.state_dict())
    print("\nTrain Seg (Fine-tuning Task 1 Encoder)...")
    trn_net(s_net, dl, 'seg', ep=20)
    
    wandb.finish()