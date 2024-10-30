# https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f

import numpy as np
from sklearn.preprocessing import scale
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.sandbox.nonparametric import kernels

# from statsmodels.stats.correlation_tools import cov_nearest

def cal_bandwidth(expr, bw_method="silverman"):

    # https://www.statsmodels.org/dev/_modules/statsmodels/nonparametric/bandwidths.html#select_bandwidth
    bw = select_bandwidth(expr, bw=bw_method, kernel=kernels.Gaussian())

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
    # import scipy.stats as stats
    # from scipy.stats import gaussian_kde
    # bw = stats.gaussian_kde(expr, bw_method = bw_method).factor

    return bw

# we can select the method 1) normal_reference, 2) scott, 3) silverman
def bandwidth_select(expr, method="silverman"):

    tot_bw = []
    for i in range(expr.shape[0]):
        try:
            res_bw = cal_bandwidth(expr[i], bw_method=method)
            tot_bw.append(res_bw)
        except Exception as e:
            print(f"Gene {i} : {str(e)}")

    median_bw = np.median([bw for bw in tot_bw if not np.isnan(bw)])

    return median_bw


# Thsi function is the same as the 'dist' function in r
def dist(coord):
    d = np.sqrt(np.sum((coord[:, np.newaxis, :] - coord[np.newaxis, :, :]) ** 2, axis=-1))
    return(d)


# please check the mellon paper.
# https://mellon.readthedocs.io/en/latest/cov.html#mellon.cov.Matern52
def cal_kernel(coord, bandwidth, method):
    
    # Exponentiated Quadratic kernel, also known as the squared exponential or the Gaussian kernel.
    if method == "gaussian":
      r = dist(coord) / bandwidth
      kernelmat = np.exp(-(r ** 2) / 2)

    # Implementation of the Matern-5/2 kernel function, a member of the Matern family of kernels.  
    elif method == "matern52":
      r = np.sqrt(5.0) * dist(coord) / bandwidth
      kernelmat = (r + (r ** 2) / 3 + 1) * np.exp(-r)
      
    # Implementation of the Matern-3/2 kernel function, a member of the Matern family of kernels.  
    elif method == "matern32":
      r = np.sqrt(3.0) * dist(coord) / bandwidth
      kernelmat = (r + 1) * np.exp(-r)
    
    # Rational Quadratic kernel function
    elif method == "ratquad":
      alpha = 1
      r = dist(coord) / bandwidth
      kernelmat = ((r ** 2) / (2 * alpha) + 1) ** -alpha
    else :
      print(" Please select 'gaussian', 'matern52', 'matern32', 'ratquad'")

    return kernelmat


def spatialkernel(expr, coord, bandwidthtype="silverman", method = "matern52", userbandwidth=None):

    # Standardization
    # print("# Scale the expression of each gene.")
    expr = scale(expr, axis=1)

    # Calculate bandwidth
    if userbandwidth is None:
        bandwidth = bandwidth_select(expr, method=bandwidthtype)
        print(f"# The bandwidth is {round(bandwidth, 4)} using {bandwidthtype}.")
    else:
        bandwidth = userbandwidth
        print(f"# The bandwidth is {round(bandwidth, 4)} (set by user).")

    # Calculate the kernel matrix using the bandwidth
    coord_normalized = scale(coord)
    kernelmat = cal_kernel(coord=coord_normalized, method=method, bandwidth=bandwidth)

    return kernelmat
