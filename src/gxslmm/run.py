import os
import shutil

import pandas as pd
import numpy as np

from cellregmap import run_association, run_interaction, estimate_betas

from .buildkernel import buildchol, spatialkernel

# import multiprocessing as mp


def gxslmm(y, x, W, S, K):

  '''
  y: outcome vector (expression phenotype, one gene only)
  x: SNP vector
  W: intercept (covariate matrix)
  S: spatial kernel 
  K: genetic relationship matrix (GRM)
  '''
  
  # TODO: W should be deconvolution results for non-cellular resolution ST data.
  if W is not None:
    W = np.asarray(W, float) 
  else:
    W = np.ones((y.shape[0], 1))

  # TODO: we need to double check the buildchol function
  hK = buildchol(K) # K = hK @ hK.T
  hS = buildchol(S) # S = hS @ hS.T

  # resutls of association and interaction test
  res_association = run_association(y=y, G=x, W=W, E=hS, hK=hK)
  res_interaction = run_interaction(y=y, G=x, W=W, E=hS, hK=hK)

  pv = pd.DataFrame({"p_association":res_association[0], "p_interaction":res_interaction[0]})  
  pd_association = pd.DataFrame(res_association[1])
  pd_association.columns = ['rho1_association', 'e2_association', 'g2_association', 'eps2_association']
  pd_interaction = pd.DataFrame(res_interaction[1])
  pd_interaction.columns = ['rho1_interaction', 'e2_interaction', 'g2_interaction', 'eps2_interaction']
  res = pd.concat([pv, pd_association, pd_interaction], axis=1)  
  return(res)

  # # estimate_betas
  # res_betas = estimate_betas(y=y, G=x, W=W, E=hS, hK=hK)
  # beta_G = res_betas[0] 
  # beta_GxC = res_betas[1][0]
  # res_beta = {'beta_G':beta_G, 'beta_GxC':beta_GxC}

  # return list[res, res_beta]