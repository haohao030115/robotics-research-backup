import torch
import torch.nn.functional as F
from va_research.models.v4.losses import objectives as base

def progress_from_q(q,initial,direction,mask):
 d=(q-initial)*direction; den=(direction*direction*mask).sum().clamp_min(1e-6)
 return ((d*mask).sum(-1)/den).clamp(0,1)
def future_progress(pred,target,initial,direction,mask):
 pp=progress_from_q(pred,initial,direction,mask);pg=progress_from_q(target,initial,direction,mask)
 hs=[5,25,49]; terms=[]
 for h in hs:
  terms.append(F.smooth_l1_loss(pp[:,h]-pp[:,0],pg[:,h]-pg[:,0]))
 return sum(terms)/len(terms),pp,pg

def objectives(pred,target,weights,initial,direction,mask):
 loss,terms=base(pred,target,[weights['body'],weights['hand'],weights['phase'],weights['progress']])
 trig=F.binary_cross_entropy_with_logits(pred['close_intent_logit'],target['close_intent'])
 adv,pp,pg=future_progress(pred['hand_ref'],target['hand_ref'],initial,direction,mask)
 total=loss+weights['trigger']*trig+weights['advance']*adv
 terms.update(trigger=trig,advance=adv)
 return total,terms,pp,pg
