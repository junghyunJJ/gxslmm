
import pandas as pd
import numpy as np
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from numpy_sugar import ddot

# from cellregmap import run_association, run_interaction, estimate_betas
from glimix_core.lmm import LMM
from glimix_core.glmm import GLMMExpFam
from tqdm import tqdm

from .buildkernel import spatialkernel
from .util import *
from .score_test import *

# import cellregmap as crm
from chiscore import davies_pvalue

# import multiprocessing as mp

class gxslmm:
  def __init__(self, y, S, X=None, hK=None):
    # expression (one gene)
    self._y = np.asarray(y, float).flatten()
    self._ncells = len(self._y)

    # covariates 
    # TODO! we need to add cell type information
    if X is not None:
      self._X = np.asarray(X, float) 
      if self._ncells != self._X.shape[0]:
        raise ValueError("Number of samples mismatch between y (gene expression) and X (covariates).")
    else:
      self._X = np.ones((self._ncells, 1))
          
    # set rnadom effects :
    # 1. hS (i.e., decomposition of spatial kernel (S = SK @ SK.T))
    # 2. hK (i.g., decomposition of GRM matrix (K = hK @ hK.T))
    if hK is None:
      self._hS = buildchol(np.asarray(S, float))
      if self._ncells != self._hS.shape[0]:
        raise ValueError("Number of samples mismatch between y (gene expression) and S (Spatial Kernel).")
    else:
      self._S = np.asarray(S, float)
      self._hS = buildchol(self._S)
      self._hK = np.asarray(hK, float)
      self._Ls = get_L_values(self._hK, self._hS) # NOTE! toooooo slow
    
    # cal QS based on the background
    self._halfSigma = {}
    self._Sigma_qs = {}
    if len(self._Ls) == 0:
      # Σ = ρ₁𝙴𝙴ᵀ
      if self._hK is None:
        self._rho1 = [1.0]
        self._halfSigma[1.0] = self._hS
        self._Sigma_qs[1.0] = economic_qs_linear(self._hS, return_q1=False)
      # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺 
      else:            
        self._rho1 = np.linspace(0, 1, 11)
        for sel_rho1 in self._rho1:
            a = np.sqrt(sel_rho1)
            b = np.sqrt(1 - sel_rho1)
            hS = np.concatenate([a * self._hS] + [b * self._hK], axis=1)
            self._halfSigma[sel_rho1] = hS
            self._Sigma_qs[sel_rho1] = economic_qs_linear(self._halfSigma[sel_rho1], return_q1=False)
    # Σ = ρ₁𝙴𝙴ᵀ + (1-ρ₁)𝙺⊙E
    else:
      self._rho1 = np.linspace(0, 1, 11)
      for sel_rho1 in self._rho1:
        a = np.sqrt(sel_rho1)
        b = np.sqrt(1 - sel_rho1)
        hS = np.concatenate([a * self._hS] + [b * L for L in self._Ls], axis=1)
        self._halfSigma[sel_rho1] = hS
        self._Sigma_qs[sel_rho1] = economic_qs_linear(self._halfSigma[sel_rho1], return_q1=False)

    
  def fsv(self, method='p'):
    '''
    Fraction of variance explained by spatial variation (FSV) 
    g == Gausian
    p == Poissan
    '''
    QS = economic_qs_linear(self._hS)
    if method == 'g':
      mm = LMM(self._y, self._X, QS, restricted=True)
    elif method == 'p': 
      mm = GLMMExpFam(self._y, 'Poisson', self._X, QS)
    mm.fit(verbose=False)

    return({"vs": [float(mm.v0)], "ve": [float(mm.v1)], "fsv":[float(mm.v0 / (mm.v0 + mm.v1))]})


  def inter(self, g, method='g'):  

    g = np.asarray(g, float)
    self._X = np.concatenate((self._X, g), axis=1)

    best = {"lml": -np.inf, "rho1": 0}
    res = {"rho1": [], "vs": [], "vg": [], "ve": [], "Q": [], "pvalue":[]}

    for sel_rho1 in tqdm(list(self._rho1)):
      QS = self._Sigma_qs[sel_rho1]

      if method == 'g':
        mm = LMM(self._y, self._X, QS, restricted=True)
      elif method == 'p': 
        mm = GLMMExpFam(self._y, 'Poisson', self._X, QS)
      mm.fit(verbose=False)
      if mm.lml() > best["lml"]:
          best["lml"] = mm.lml()
          best["rho1"] = sel_rho1
          best["mm"] = mm
  
    mm = best["mm"]
    K0 = mm.covariance()
    P0 = cal_P0(self._X, K0)
    Q = cal_Q(P0, self._S, g, self._y)
    pvalue = cal_p(Q, P0, g, self._hS)

    res["rho1"].append(best["rho1"])
    res["vs"].append(mm.v0 * best["rho1"])
    res["vg"].append(mm.v0 * (1 - best["rho1"]))
    res["ve"].append(mm.v1)
    res["Q"].append(Q)
    res["pvalue"].append(pvalue)
    df_res = pd.DataFrame(res)

    return(df_res)


  def assoc(self, G, method='g'):

    res = {"rho1": [], "vs": [], "vg": [], "ve": [], "null_lml": [], "alt_lml": [], "pvalue": []}    
    best = {"lml": -np.inf, "rho1": 0}

    for sel_rho1 in tqdm(list(self._rho1)):

      QS = self._Sigma_qs[sel_rho1]
      if method == 'g':
        mm = LMM(self._y, self._X, QS, restricted=False)
      elif method == 'p': 
        mm = GLMMExpFam(self._y, 'Poisson', self._X, QS)
      mm.fit(verbose=False)

      if mm.lml() > best["lml"]:
          best["lml"] = mm.lml()
          best["rho1"] = sel_rho1
          best["mm"] = mm

    null_mm = best["mm"]
    res["rho1"].append(best["rho1"])
    res["vs"].append(null_mm.v0 * best["rho1"])
    res["vg"].append(null_mm.v0 * (1 - best["rho1"]))
    res["ve"].append(null_mm.v1)
    res["null_lml"].append(null_mm.lml())

    n_snps = G.shape[1]
    for i in tqdm(range(n_snps)):
      g = G[:, [i]]
      X = np.concatenate((self._X, g), axis=1)
      QS = self._Sigma_qs[best["rho1"]]

      if method == 'g':
        alt_mm = LMM(self._y, self._X, QS, restricted=False)
      elif method == 'p': 
        alt_mm = GLMMExpFam(self._y, 'Poisson', self._X, QS)
      alt_mm.fit(verbose=False)
      
      alt_lml = alt_mm.lml()
      res["alt_lml"].append(alt_lml)
      pvalue = lrt_pvalues(null_mm.lml(), alt_lml, dof=1)
      res["pvalue"].append(pvalue)

    return pd.DataFrame(res)
