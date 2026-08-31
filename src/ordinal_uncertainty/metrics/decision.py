"""Exact discrete ordinal Bayes decisions and their predictive risks."""
from __future__ import annotations
import numpy as np
def bayes_decisions(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    p=np.asarray(probabilities,float); classes=np.arange(p.shape[1]); actions=classes[None,:,None]; outcomes=classes[None,None,:]
    l1=(p[:,None,:]*np.abs(outcomes-actions)).sum(2); l2=(p[:,None,:]*(outcomes-actions)**2).sum(2)
    mode=p.argmax(1)
    return {'mode_decision':mode,'l1_bayes_decision':l1.argmin(1),'l2_bayes_decision':l2.argmin(1),'mode_l1_risk':l1[np.arange(len(p)),mode],'l1_bayes_risk':l1.min(1),'l2_bayes_risk':l2.min(1)}
