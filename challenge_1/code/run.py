# %%
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cdist, pdist, squareform

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
            'Bach2'     : ['Lef1','Dvl3' , 'Tcf7', 'z-NOSITE', 'z1-NONGENE'],
            'Bhlhe40'   : ['Hmgb1','Hmgb2','Id2','Litaf','Nr4a2','Sub1','Tpt1'],
            'Ets1'      : ['Hif1a'],
            'Fosb'      : ['Il12rb1','Prdm1','Tcf7', 'Lef1'],
            'Mafk'      : ['Lef1','Dvl3','Oxnad1','Dvl3','Tcf7' ,'Il12rb1','Prdm1'],
            'Stat3'     : ['Id2','Nr4a2' ,  'Litaf'],
            }
print('Closest neighbors in expression space:')
print(neighbors)
# %%

goi = ['Stat4'] + list(neighbors.keys()) + list(set(itertools.chain(*neighbors.values())))
goi = sorted(goi)
print(goi)
expr_df = sc.get.obs_df(adata, keys=sorted(list(ko_genes.union(goi).intersection(adata.var_names))))
#px.scatter(expr_df.join(adata.obs).fillna(0), color='state', 
#            y='Dvl2', x='Dvl3', #y='Hmgb1',x='Hmgb2', #y='Stat3', x='Stat4', #y='Ets1', x='Hif1a', 
#            hover_name='condition', opacity=0.25, height=600)
#%%
# test_gene = 'Dvl3'
# train_gene = 'Dvl2'
# expr0count = expr_df.query(f'{test_gene} < 0.1 & {train_gene} > 0.1').join(adata.obs).fillna(0).groupby('state').count()[test_gene]
# expr0frac = expr0count / expr0count.sum()
# pd.concat([expr0count, expr0frac, cond_csf[train_gene]], axis=1)
#%%
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
#csc
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
#cond_csf = cond_csf.div(cond_csf.sum(axis=1), axis=0)
cond_csf = cond_csf.T
#px.imshow(cond_csf).show()
nosite_csf = cond_csf.loc[:,'z-NOSITE']
nosite_csf = nosite_csf.div(nosite_csf.sum(axis=0))
nongene_csf = cond_csf.loc[:,'z1-NONGENE']
nongene_csf = nongene_csf.div(nongene_csf.sum(axis=0))
control_csf = pd.concat([nosite_csf, nongene_csf], axis=1)
# %%
target_csf = csf.reset_index().groupby('gRNA_maxID').sum()
#target_csf = target_csf.div(target_csf.sum(axis=1), axis=0)
target_csf = target_csf.T
#px.imshow(target_csf).show()
nositeT_csf = target_csf.filter(like='NOSITE')
nosite_csf = nositeT_csf.div(nositeT_csf.sum(axis=0))
nongeneT_csf = target_csf.filter(like='NONGENE')
nongeneT_csf = nongeneT_csf.div(nongeneT_csf.sum(axis=0))
controlT_csf = pd.concat([nositeT_csf, nongeneT_csf], axis=1)
# %%
def prep_out(genes, truth=None):
#try:
#    genes = val_genes+test_genes
    l1 = pd.Series(index=genes)
    df = pd.DataFrame(index = cond_csf.index)
    truth = truth.copy()
    truth.index = [g+'_true' for g in truth.index]
    truth.columns = ['progenitor','effector','terminal exhausted','cycling','other']
    for gene in genes:
        closest = neighbors[gene][0]
        close0count = expr_df.query(f'{closest} < 0.1 & {gene} > 0.1').join(adata.obs).fillna(0).groupby('state').count()[closest]
        close0frac = close0count / close0count.sum()
        close0frac.name = closest+'-expr0'
        expr0count = expr_df.query(f'{gene} < 0.1 & {closest} > 0.1').join(adata.obs).fillna(0).groupby('state').count()[gene]
        expr0frac = expr0count / expr0count.sum()
        expr0frac.name = gene+'-expr0'
        loc_csc = cond_csf.loc[:, neighbors[gene]]
        sum_csc = loc_csc.sum(axis=1)
        sum_csc.name = gene+'~KO'
        loc_csf = loc_csc.div(loc_csc.sum(axis=0)).round(5)
        df[gene] = sum_csc.div(sum_csc.sum()).round(5)
        l1[gene] = np.abs(loc_csf.T - df[gene]).sum(axis=1).median(axis=0)
        if truth is None:
            px.imshow(pd.concat([close0frac, loc_csf, df[gene], expr0frac, control_csf], axis=1)).show() ####
        else:
            px.imshow(pd.concat([close0frac, loc_csf, df[gene], truth.loc[gene+'_true'], expr0frac, control_csf], axis=1)).show() ####

    df.index = ['d_i','b_i','e_i','a_i','c_i']
    df = df.sort_index().T
    df.index.name = 'gene'
    l1.name = 'L1_expected'
    return df, l1
#except ValueError as err:
#    print(err)

df, l1 = prep_out(val_genes+test_genes, truth=pd.concat([true_val, true_test], axis=0))
print('Expected L1 loss for each validation/test gene:')
print(l1)

# %%
# %%
# Output predictions on validation genes
val_out,val_l1 = prep_out(val_genes)
val_out.to_csv('../solution/validation_output.csv')
print(val_out)
print('Expected avg. validation L1 loss:', val_l1.mean().round(5))
# %%
true_val = val_out.copy()
true_val.loc['Aqr'] = [0.026548673, 0.725663717, 0.115044248, 0.115044248, 0.017699115]
true_val.loc['Bach2'] = [0.010309278, 0.175257732, 0.333333333, 0.463917526, 0.017182131]
true_val.loc['Bhlhe40'] = [0.9375, 0.03125, 0.03125, 0, 0]
l1_val = (val_out - true_val).abs().sum(axis=1)
l1_val.name = 'L1_actual'
js_val = pd.Series(cdist(true_val, val_out, metric='jensenshannon').diagonal(), index=l1_val.index) 
js_val.name = 'JS_actual'
print(f'Actual avg. validation L1 loss: {l1_val.mean().round(5)} and JS dist: {js_val.mean().round(5)}' )
pd.concat([val_l1, l1_val, js_val], axis=1)
# %%
# Output predictions on test genes
test_out,test_l1 = prep_out(test_genes)
test_out.to_csv('../solution/test_output.csv')
print(test_out)
print('Expected avg. test L1 loss:', test_l1.mean().round(5))
# %%
true_test = test_out.copy()
true_test.loc['Ets1'] = [0.02016129, 0.008870968, 0.02983871, 0.046774194, 0.894354839]
true_test.loc['Fosb'] = [0.355371901, 0.082644628, 0.231404959, 0.32231405, 0.008264463]
true_test.loc['Mafk'] = [0.337078652, 0.415730337, 0.06741573, 0.168539326, 0.011235955]
true_test.loc['Stat3'] = [0.042253521, 0.126760563, 0.338028169, 0.464788732, 0.028169014]
l1_test = (true_test - test_out).abs().sum(axis=1) 
l1_test.name = 'L1_actual'
js_test = pd.Series(cdist(true_test, test_out, metric='jensenshannon').diagonal(), index=l1_test.index) 
js_test.name = 'JS_actual'
print(f'Actual avg. test L1 loss: {l1_test.mean().round(5)} and JS dist: {js_test.mean().round(5)}' )
pd.concat([test_l1, l1_test, js_test], axis=1)

# %%
