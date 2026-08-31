import numpy as np
from ordinal_uncertainty.metrics.decision import bayes_decisions
def test_discrete_bayes_decisions_and_risks():
 d=bayes_decisions(np.array([[0,0,1,0,0.],[0,.5,.5,0,0.],[.5,0,0,0,.5]]))
 assert d['mode_decision'].tolist()==[2,1,0]
 assert d['l1_bayes_decision'].tolist()==[2,1,0]
 assert d['l2_bayes_decision'].tolist()==[2,1,2]
 assert d['mode_l1_risk'][0]==d['l1_bayes_risk'][0]==d['l2_bayes_risk'][0]==0
