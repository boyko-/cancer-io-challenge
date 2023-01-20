# %%
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# known good versions:
# scanpy==1.9.1 anndata==0.8.0 umap==0.5.3 numpy==1.23.4 scipy==1.9.3 pandas==1.5.1 scikit-learn==1.1.2 statsmodels==0.13.2 python-igraph==0.10.2 pynndescent==0.5.7

# %%
datadir = '/home/boyko/data/cancer-io-challenge'
adata = sc.read_h5ad(f"{datadir}/sc_training.h5ad")
adata.obs.index.name = 'cell'
adata.obs.info()
# %%
## adata.X stores the stadard-normalized data, but you can also reconstruct it using adata.layers['rawcounts']

# make a copy of normalized logcounts
adata.layers['normalized_logcounts'] = adata.X.copy()
# standard normalization
adata.X = adata.layers['rawcounts'].copy()
sc.pp.normalize_total(adata, target_sum=5e3)
sc.pp.log1p(adata)
# %%
# Semi-manual processing of sc_training data is done separately in scratch.py
# The resulting map between each validation/test gene and its K closest neighbors
# is hard-coded below, but final mapping will tune variable K on validation genes    
# %%
ko_genes = set(adata.obs.condition)
val_genes = ['Aqr','Bach2','Bhlhe40']   
test_genes = ['Ets1','Fosb','Mafk','Stat3']
neighbors = {'Aqr'      : ['Myb','Eomes' , 'Il12rb2','Tcf3'],
            'Bach2'     : ['Lef1','Dvl2', 'Dvl3' , 'Tcf7', 'z-NOSITE', 'z1-NONGENE'],
            'Bhlhe40'   : ['Eef2','Hmgb1','Hmgb2','Id2','Litaf','Nr4a2','Rps6','Sub1','Tpt1'],
            'Ets1'      : ['Hif1a'],
            'Fosb'      : ['Il12rb1','Klf2','Prdm1','Tcf7', 'Lef1','Foxm1'],
            'Mafk'      : ['Lef1','Dvl2','Dvl3','Oxnad1','Dvl3','Tcf7' ,  'Il12rb1','Klf2','Prdm1'],
            'Stat3'     : ['Sub1','Id2','Nr4a2' ,  'Litaf'],
            }
print('Closest neighbors in expression space:')
print(neighbors)
# %%
import itertools
goi = list(neighbors.keys()) + list(set(itertools.chain(*neighbors.values())))
goi = sorted(goi)
print(goi)
expr_df = sc.get.obs_df(adata, keys=sorted(list(ko_genes.union(goi).intersection(adata.var_names))))
#px.scatter(expr_df.join(adata.obs).fillna(0), color='state', 
#            y='Stat3', x='Stat4', #y='Ets1', x='Hif1a', 
#            hover_name='condition', opacity=0.25, height=600)
#%%
from scipy.spatial.distance import pdist, squareform
nG = expr_df.shape[1]
pairwise = pd.DataFrame(
    squareform(pdist(expr_df.T, 
        metric='jensenshannon')), # + np.eye(nG),
    columns = expr_df.columns,
    index = expr_df.columns
)
#px.imshow(pairwise, height=800, title='scRNA-seq distance between each pair of KO genes')
# %%
csc = adata.obs.query(f"condition in {goi} | condition == 'Unperturbed' ")
csc = csc.groupby(['condition','gRNA_maxID','state']).count()
csc = csc.query('lane > 0').sort_index()
csc = csc.unstack().fillna(0)
csc.columns = csc.columns.get_level_values('state').to_list()
csc.sort_index(inplace=True)
csc
# %%
csf = csc.copy()
conds = csc.index.get_level_values('condition').to_list()
num_unperturbed = sum(['Unperturbed' == _ for _ in conds])
grnas = csc.index.get_level_values('gRNA_maxID').to_list()
num_nosite = sum([_.startswith('NO-SITE') for _ in grnas])
num_nongene = sum(['NON-GENE' in _ for _ in grnas])
conds = conds[:-num_unperturbed] + ['z-NOSITE'] * num_nosite + ['z1-NONGENE'] * num_nongene
csf.index = pd.MultiIndex.from_arrays([conds,grnas], names=['condition','gRNA_maxID'])
csf.sort_index(inplace=True)

# %%
cond_csf = csf.reset_index().groupby('condition').sum()
cond_csf = cond_csf.div(cond_csf.sum(axis=1), axis=0).T
#px.imshow(cond_csf).show()
nosite_csf = cond_csf.loc[:,'z-NOSITE']
nongene_csf = cond_csf.loc[:,'z1-NONGENE']
control_csf = pd.concat([nosite_csf, nongene_csf], axis=1)
# %%
target_csf = csf.reset_index().groupby('gRNA_maxID').sum()
target_csf = target_csf.div(target_csf.sum(axis=1), axis=0).T
#px.imshow(target_csf).show()
nositeT_csf = target_csf.filter(like='NOSITE')
nongeneT_csf = target_csf.filter(like='NONGENE')
controlT_csf = pd.concat([nosite_csf, nongene_csf], axis=1)
# %%
def prep_out(genes):
#try:
#    genes = val_genes+test_genes
    l1 = pd.Series(index=genes)
    df = pd.DataFrame(index = cond_csf.index)

    for gene in genes:
        loc_csf = cond_csf.loc[:, neighbors[gene]]
        mean_csf = loc_csf.mean(axis=1)
        mean_csf.name = gene
        df[gene] = mean_csf.div(mean_csf.sum()).round(5)
        l1[gene] = np.abs(loc_csf.T - mean_csf).sum(axis=1).median(axis=0)
#        px.imshow(pd.concat([loc_csf, mean_csf, control_csf], axis=1)).show() ####
    df.index = ['d_i','b_i','e_i','a_i','c_i']
    df = df.sort_index().T
    df.index.name = 'gene'
    return df, l1
#except ValueError as err:
#    print(err)

df, l1 = prep_out(val_genes+test_genes)
print('Expected L1 loss for each validation/test gene:')
print(l1)
# %%
# Output predictions on validation genes
val_out,val_l1 = prep_out(val_genes)
val_out.to_csv('../solution/validation_output.csv')
print(val_out)
print('Expected avg. validation L1 loss:', val_l1.mean().round(5))
# %%
# Output predictions on test genes
test_out,test_l1 = prep_out(test_genes)
test_out.to_csv('../solution/test_output.csv')
print(test_out)
print('Expected avg. test L1 loss:', test_l1.mean().round(5))
# %%
