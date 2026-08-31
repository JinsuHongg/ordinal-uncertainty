"""CORAL output conversion and differentiable RPS loss."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
 return (labels[:,None] > torch.arange(num_classes-1,device=labels.device)).float()
def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
 return F.binary_cross_entropy_with_logits(logits,coral_targets(labels,logits.shape[1] + 1),reduction='mean')
def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
 q=torch.sigmoid(logits); p=torch.cat([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1)
 if torch.any(p < -1e-6): raise ValueError('CORAL cumulative probabilities are not monotone; refusing to repair them')
 return p.clamp_min(0) / p.clamp_min(0).sum(1,keepdim=True)
def coral_prediction(logits: torch.Tensor) -> torch.Tensor:
 """Official CORAL threshold-count prediction (sigmoid(logit) > .5)."""
 return (torch.sigmoid(logits) > .5).sum(1)
def rps_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
 p=F.softmax(logits,1); target=F.one_hot(labels,num_classes=logits.shape[1]).float()
 return ((p.cumsum(1)[:,:-1]-target.cumsum(1)[:,:-1])**2).sum(1).mean()/(logits.shape[1]-1)
class CoralHead(nn.Module):
 def __init__(self,features:int,num_classes:int): super().__init__();self.weight=nn.Linear(features,1,bias=False);self.bias=nn.Parameter(torch.zeros(num_classes-1))
 def forward(self,x): return self.weight(x)+self.bias
