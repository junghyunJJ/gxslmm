import numpy as np
import pandas as pd
import xarray as xr

# from numpy.linalg import cholesky
from .util import buildchol

def expand_grm(sample_mapping, df_K):
  '''
  The sample_mapping should have the following columns:
    1st column is the donor id 
    2nd column is the cell id 
  '''

  # extract unique individuals
  cells = sample_mapping.iloc[:, 1].unique() # cell names
  donors = sample_mapping.iloc[:, 0].unique() # individual names
  donors.sort()

  print(f'# of individual: {len(donors)} / # of cells: {len(cells)}')

  assert all(df_K.columns == df_K.index) #symmetric matrix, donors x donors
  K = xr.DataArray(df_K.values, dims=["sample_0", "sample_1"], coords={"sample_0": df_K.columns, "sample_1": df_K.index})
  K = K.sortby("sample_0").sortby("sample_1")
  donors = sorted(set(list(K.sample_0.values)).intersection(donors))
  print("Number of donors after GRM intersection: {}".format(len(donors)))


  # decompose such as K = hK @ hK.T (using Cholesky decomposition)
  # hK = cholesky(K.values)
  hK = buildchol(K.values)
  
  hK = xr.DataArray(hK, dims=["sample", "col"], coords={"sample": K.sample_0.values})
  assert all(hK.sample.values == K.sample_0.values)

  ## use sel from xarray to expand hK (using the sample mapping file)
  hK_expanded = hK.sel(sample=sample_mapping.iloc[:, 0].values)
  assert all(hK_expanded.sample.values == sample_mapping.iloc[:, 0].values)
  df_hK_expanded = pd.DataFrame(hK_expanded.values)
  df_hK_expanded.index = list(sample_mapping.iloc[:, 0])
  df_hK_expanded
  # df_hK_expanded.to_numpy()
  return(df_hK_expanded)

def expand_geno(sample_mapping, df_geno):
  '''
  The df_geno = snp x donors 
  '''

  # extract unique individuals
  snps = df_geno.index.unique() # snps names
  cells = sample_mapping.iloc[:, 1].unique() # cell names
  donors = sample_mapping.iloc[:, 0].unique() # unique individual names
  donors.sort()

  # check geno file donor with sample_mapping donor
  geno_donors = list(df_geno.columns.unique())
  geno_donors.sort()
  assert all( np.array(geno_donors) == donors)


  print(f'# of individual: {len(donors)} / # of cells: {len(cells)} / # of snps: {len(snps)}') 

  # geno = xr.DataArray(df_geno.values, dims=["sample_0", "sample_1"], coords={"sample_0": df_geno.columns, "sample_1": df_geno.index})
  geno = xr.DataArray(df_geno.values, dims=["sample_0", "sample_1"], coords={"sample_0": df_geno.index, "sample_1": df_geno.columns})

  ## use sel from xarray to expand hK (using the sample mapping file)
  geno_expanded = geno.sel(sample_1=sample_mapping.iloc[:, 0].values)
  assert all(geno_expanded.sample_1.values == sample_mapping.iloc[:, 0].values)
  geno_expanded = pd.DataFrame(geno_expanded.values)
  geno_expanded.columns = list(sample_mapping.iloc[:, 0])
  geno_expanded.index = df_geno.index
  # geno_expanded.to_numpy()
  return(geno_expanded)
