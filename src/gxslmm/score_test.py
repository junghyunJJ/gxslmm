import numpy as np
from numpy.linalg import inv
from numpy_sugar import ddot
from chiscore import davies_pvalue

# https://github.com/limix/CellRegMap/blob/main/cellregmap/_cellregmap.py
# https://github.com/limix/CellRegMap/blob/main/cellregmap/_math.py
# https://github.com/limix/struct-lmm/blob/master/struct_lmm/_lmm.py

def cal_P0(X, K0):
  invK0 = inv(K0)
  P0 = invK0 - invK0 @ X @ inv(X.T @ invK0 @ X) @ X.T @ invK0
  return(P0)

def cal_Q(P0, S, g, y):
  g = g.ravel()
  deltaK = np.diag(g) @ S @ np.diag(g)
  Q = 0.5 * (y.T @ P0 @ deltaK @ P0 @ y)
  return(Q)

def cal_p(Q, P0, g, hS):
  g = g.ravel()
  _sqrt_dK = ddot(g, hS)
  matrix_for_dist_weights = 0.5 * _sqrt_dK.T @ P0.dot(_sqrt_dK)

  pval, pinfo = davies_pvalue(Q, matrix_for_dist_weights, True)
  return(pval)


# # K0 = mm.covariance()
# # P0 = inv(K0) - inv(K0) @ X @ inv(X.T @ inv(K0) @ X) @ X.T @ inv(K0)

# # EE = E0 @ E0.T
# gtest = g.ravel()
# deltaK = np.diag(gtest) @ EE @ np.diag(gtest)
# # _sqrt_dK = ddot(gtest, E0)
# # deltaK = _sqrt_dK @ _sqrt_dK.T

# Q_ = 0.5 * (self._y.T @ P0 @ deltaK @ P0 @ self._y)


# _sqrt_dK = ddot(gtest, E0)
# matrix_for_dist_weights = 0.5 * _sqrt_dK.T @ P0.dot(_sqrt_dK)

# from chiscore import davies_pvalue
# pval, pinfo = davies_pvalue(Q, matrix_for_dist_weights, True)
