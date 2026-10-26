import torch
import torch.nn.functional as F
from va_research.models.v4.losses import objectives as base_objectives

def objectives(pred, target, weights):
    base, terms=base_objectives(pred,target,[weights.get('body',1.),weights.get('hand',1.),weights.get('phase',.1),weights.get('progress',.1)])
    ready=(target['phase']>=1).float()
    trigger=F.binary_cross_entropy_with_logits(pred['grasp_ready_logit'][:,0],ready)
    vel_t=target['hand_ref'][:,1:]-target['hand_ref'][:,:-1]
    vel_p=pred['hand_ref'][:,1:]-pred['hand_ref'][:,:-1]
    velocity=F.smooth_l1_loss(vel_p,vel_t)
    # Only penalize reversals where the source label is closing/hold and the
    # expert itself does not reopen over that local interval.
    active=(target['phase']>=1)[:,None,None]
    reopening=torch.relu(-(vel_p*active)).mean()
    total=base+weights.get('trigger',.1)*trigger+weights.get('velocity',.1)*velocity+weights.get('reopen',.02)*reopening
    terms.update(trigger=trigger,velocity=velocity,reopen=reopening)
    return total,terms
