import argparse as ap
import sys

from numpy import ones
from numpy.random import RandomState
import gxslmm as gs


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-n", default=50, type=int, help="# number of samples (cells)")
    argp.add_argument("-p", default=5, type=int, help="# number of individuals")
    argp.add_argument("--seed", default=12345, type=int, help="Random seed")

    args = argp.parse_args(args)
    random = RandomState(args.seed)
    n = args.n                           # number of samples (cells)
    p = args.p                           # number of individuals
    y = random.randn(n, 1)               # outcome vector (expression phenotype, one gene only)
    W = ones((n, 1))                     # intercept (covariate matrix)
    hS = random.randn(n, p)              # decomposition of Spatial kernel (S = hS @ hS.T)
    
    random = RandomState(args.seed + 1)
    hK = random.randn(n, p)              # decomposition of kinship matrix (K = hK @ hK.T)
    x = 1.0 * (random.rand(n, 1) < 0.2)  # SNP vector
    
    S = hS @ hS.T
    K = hK @ hK.T
    
    res = gs.gxslmm(y = y, x = x, W = W, S = S, K = K)
    print(res)
    # import pdb; pdb.set_trace()

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))