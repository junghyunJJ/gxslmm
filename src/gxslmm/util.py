# https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
import numpy as np
from numpy import linalg as la


def nearestPD(A):
  """Find the nearest positive-definite matrix to input

  A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
  credits [2].

  [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

  [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
  matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
  """

  B = (A + A.T) / 2
  _, s, V = la.svd(B)

  H = np.dot(V.T, np.dot(np.diag(s), V))

  A2 = (B + H) / 2

  A3 = (A2 + A2.T) / 2

  if isPD(A3):
      return A3

  spacing = np.spacing(la.norm(A))
  # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
  # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
  # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
  # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
  # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
  # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
  # `spacing` will, for Gaussian random matrixes of small dimension, be on
  # othe order of 1e-16. In practice, both ways converge, as the unit test
  # below suggests.
  identity = np.eye(A.shape[0])
  k = 1
  while not isPD(A3):
      mineig = np.min(np.real(la.eigvals(A3)))
      A3 += identity * (-mineig * k**2 + spacing)
      k += 1

  return A3


def isPD(B):
  """Returns true when input is positive-definite, via Cholesky"""
  try:
      _ = la.cholesky(B)
      return True
  except la.LinAlgError:
      return False


def buildchol(skernel):
  if isPD(skernel) is False:
      # skernel_nearPD = cov_nearest(skernel) # too slow
      skernel_nearPD = nearestPD(skernel)
      chol_skernel = np.linalg.cholesky(skernel_nearPD)
  else:
      chol_skernel = np.linalg.cholesky(skernel)

  return chol_skernel


def get_L_values(hK, E):
    from numpy_sugar.linalg import economic_svd
    from numpy_sugar import ddot

    """
    As the definition of Ls is not particulatly intuitive,
    function to extract list of L values given kinship K and 
    cellular environments E
    """
    # get eigendecomposition of EEt
    [U, S, _] = economic_svd(E)
    us = U * S

    # get decomposition of K \odot EEt
    Ls = [ddot(us[:,i], hK) for i in range(us.shape[1])]
    return Ls

def lrt_pvalues(null_lml, alt_lmls, dof=1):
    """
    Compute p-values from likelihood ratios.

    These are likelihood ratio test p-values.

    Parameters
    ----------
    null_lml : float
        Log of the marginal likelihood under the null hypothesis.
    alt_lmls : array_like
        Log of the marginal likelihoods under the alternative hypotheses.
    dof : int
        Degrees of freedom.

    Returns
    -------
    pvalues : ndarray
        P-values.
    """
    from numpy import clip
    from numpy_sugar import epsilon
    from scipy.stats import chi2

    lrs = clip(-2 * null_lml + 2 * np.asarray(alt_lmls, float), epsilon.super_tiny, np.inf)
    pv = chi2(df=dof).sf(lrs)
    return clip(pv, epsilon.super_tiny, 1 - epsilon.tiny)

def cis_snp_selection(feature_id, annotation_df, G, window_size):
  # https://github.com/annacuomo/CellRegMap_analyses/blob/main/endodiff/preprocessing/Expand_genotypes_kinship.ipynb

  # gene_name = "ENSG00000001617" # (e)gene name
  # w = 100000 # window size (cis)
  # anno_df = annotation linking gene to genomic position

  # G_sel = cis_snp_selection(gene_name, anno_df, G, w)

  feature = annotation_df.query("feature_id==\"{}\"".format(feature_id)).squeeze()
  chrom = str(feature['chromosome'])
  start = feature['start']
  end = feature['end']
  # make robust to features self-specified back-to-front
  lowest = min([start,end])
  highest = max([start,end])
  # for cis, we sequentially add snps that fall within each region
  G = G.where((G.chrom == str(chrom)) & (G.pos > (lowest-window_size)) & (G.pos < (highest+window_size)), drop=True)
  return G
