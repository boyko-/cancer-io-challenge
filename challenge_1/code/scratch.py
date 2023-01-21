# %%
# Considering the over-arching goal of using the results from all 3 Challenges to pick new genes for knockout and assay in Challenge 2, I would suggest augmenting/replacing the L1 loss with a scoring function that accounts for uncertainty in both the prediction and in the observed distribution of states, like KL divergence.
# To illustrate this, I'll use the scoring example from Challenge 1 Overview, where scoring with L1 loss = 0.06 between the observed and predicted proportions assumes equal uncertinaty among the 5 proportions, so any other prediction with the same L1 loss would be considered equally good -- including (0.195, 0.105, 0.295, 0.040, 0.405) which can be interpreted as either slightly better for predicting the non-zero proportions of the "true"/observed distribution even closer, or  significantly worse for giving even 4% weight for the truly-zero proportion
# However considering the large variance of state distributions among individual gRNAs targeting the same knockout (for genes like , I  
# %%
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px
from shannonca.dimred import reduce_scanpy
from collections import Counter
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
print(adata.obs.info())
adata
# known result:
# AnnData object with n_obs × n_vars = 28697 × 15077
#    obs: 'gRNA_maxID', 'state', 'condition', 'lane', 
# layers: 'rawcounts'
#plt.hist(np.abs(bdata.X[g2i['Aqr']] - bdata.X[g2i['Myb']]), bins=range(-100,100,1)) 
# %%
val_genes = ['Aqr','Bach2','Bhlhe40']
test_genes = ['Ets1','Fosb','Mafk','Stat3']
#goi = ['Batf','Ctla4','Ctnnb1','Eomes','Ep300','Ezh2','Foxm1','Gzma','Il12rb1','Id2','Id3','Ifng','Irf2','Irf9','Klf2','Myb','Oct4','Pdcd1','Prdm1','Prf1','Runx2','Sox4','Stat4','Tcf3','Tcf7','Tox','Tox2','Unperturbed']
goi = ['Bhlhe40','Stat3', 'Sub1','Id2','Nr4a2','Litaf'] #['Aqr', 'Myb','Eomes','Il12rb2','Tcf3'] #,  'Bach2','Lef1','Dvl2','Oxnad1','Dvl3','Tcf7',  'Bhlhe40','Sub1','Id2','Nr4a2','Litaf' ]   #'Lef1','Prdm1','Foxm1','Klf2','Tcf7','Sp140','Ep300','Foxp1','Hif1a','Sp100','Crem','Litaf','Nr4a2','Id2','Sub2','Hmgb1','Hmgb2']
ko_genes = set(adata.obs.condition)
print(goi) #sorted(goi))

# %%
csc = adata.obs.query(f"condition in {goi} | condition == 'Unperturbed' ")
csc = csc.groupby(['condition','gRNA_maxID','state']).count()
csc = csc.query('lane > 0').sort_index()
csc = csc.unstack().fillna(0)
csc.columns = csc.columns.get_level_values('state').to_list()
csc.sort_index(inplace=True)
csc.describe()

# %%
#nosite = pd.MultiIndex.from_product([['Unperturbed'], csc.index.get_level_values('gRNA_maxID')[csc.index.get_level_values('gRNA_maxID').str.startswith('NO-SITE')]], names=['condition','gRNA_maxID']).to_flat_index()
#nongene = pd.MultiIndex.from_product([['Unperturbed'], csc.index.get_level_values('gRNA_maxID')[csc.index.get_level_values('gRNA_maxID').str.contains('NON-GENE')]], names=['condition','gRNA_maxID'])

# %%
csf = csc.copy()
conds = csc.index.get_level_values('condition').to_list()
num_unperturbed = sum(['Unperturbed' == _ for _ in conds])
grnas = csc.index.get_level_values('gRNA_maxID').to_list()
num_nosite = sum([_.startswith('NO-SITE') for _ in grnas])
num_nongene = sum(['NON-GENE' in _ for _ in grnas])
conds = conds[:-num_unperturbed] + ['z-NOSITE'] * num_nosite + ['z1-NONGENE'] * num_nongene
csf.index = pd.MultiIndex.from_arrays([conds,grnas], names=['condition','gRNA_maxID'])
csf = csf.sort_index()
display(csf.iloc[:-num_unperturbed,:])
# %%
cond_csf = csf.reset_index().groupby('condition').sum()
#display(
px.imshow(cond_csf.div(cond_csf.sum(axis=1), axis=0).T)

# %%
target_csf = csf.reset_index().groupby('gRNA_maxID').sum()
#display(
px.imshow(target_csf.div(target_csf.sum(axis=1), axis=0).T) 
#            height=1000, color_continuous_scale="Viridis")

# %%
# establish expression (and later accessibility) profile similarity among genes:
# 1) Transpose the expression AnnData matrix X
# 2) Run PCA on X.T (which also computes the eigen values of each )
# 3) Find 50 nearest neighbors for each gene 
# %%
## adata.X stores the stadard-normalized data, but you can also reconstruct it using adata.layers['rawcounts']

# make a copy of normalized logcounts
adata.layers['normalized_logcounts'] = adata.X.copy()
# standard normalization
adata.X = adata.layers['rawcounts'].copy()
sc.pp.normalize_total(adata, target_sum=5e3)
sc.pp.log1p(adata)

# %%
## visualize cell state clusters on umap
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
sc.tl.umap(adata, min_dist=0.3)
# known result:
# computing PCA | with n_comps=50 | finished (0:00:32)
# computing neighbors using 'X_pca' | with n_pcs = 50 | finished: added to `.uns['neighbors']`
#    `.obsp['distances']`, distances for each pair of neighbors
#    `.obsp['connectivities']`, weighted adjacency matrix (0:00:20)
# computing UMAP | finished: added
#    'X_umap', UMAP coordinates (adata.obsm) (0:00:23) 

# %%
sc.pl.umap(adata, color=['condition'], legend_loc='right margin')
sc.pl.umap(adata, color=['state'], palette='Accent', legend_loc='right margin')
# compare to known UMAP plot
# %%
sc.tl.tsne(adata)
sc.pl.tsne(adata, color=['condition'], legend_loc='right margin', )
sc.pl.tsne(adata, color=['state'], palette='Accent', legend_loc='right margin')
# %%



# %%
bdata = adata.transpose()
bdata
# %%
sc.tl.pca(bdata, svd_solver='arpack')
sc.pp.neighbors(bdata, n_neighbors=15, n_pcs=50)
sc.tl.umap(bdata, min_dist=0.3)
# %%
from collections import Counter
bdata.obs['KO_gene'] = ['val' if g in val_genes else 'test' if g in test_genes else 'train' if g in ko_genes else '???' for g in bdata.obs.index]
Counter(bdata.obs.KO_gene)
# %%
sc.pl.umap(bdata, color=['KO_gene'], legend_loc='right margin', alpha=0.5, palette='')
bdata.obs['KO_gene'] = ['z???' if _ == '???' else _ for _ in bdata.obs.KO_gene]
sc.pl.umap(bdata, color=['KO_gene'], legend_loc='right margin', alpha=0.5, palette='Accent')
# %%

# %%
conn_csr = bdata.uns['neighbors']['connectivities']
dist_csr = bdata.uns['neighbors']['distances']
g2i = pd.Series(range(len(bdata.obs_names)), index=bdata.obs_names)

for g_str in val_genes:
    g = g2i[g_str]
    conn_g = conn_csr[g].nonzero()[:][1]
    conn_gstr = bdata.obs_names[conn_g]
    conn_s = pd.Series(conn_csr[g, conn_g].toarray().flat, index=conn_gstr)
    dist_s = pd.Series(dist_csr[g, conn_g].toarray().flat, index=conn_gstr)
    closest = conn_s.index[conn_s >= 1.0]
    print(g_str,' conn/dist with:\n', pd.concat([conn_s[closest], dist_s[closest]], axis=1, keys=['conn','dist']))
    print()

# %%
closest_gene = bdata.uns['neighbors']['connectivities'].argmax(axis=1)
closest_idx = closest_gene[bdata.obs_names == 'Aqr']
bdata.obsm['X_umap'][closest_idx], bdata.obs_names[closest_idx]
# %%
known_gene_bin = [g in ko_genes for g in bdata.obs_names]
closest_known_idx = bdata.uns['neighbors']['connectivities'][known_gene_bin].argmax(axis=0)
closest_known_gene = pd.Series(bdata.obs_names[closest_known_idx.flat], index=bdata.obs_names)
closest_known_gene

# %%
umap_df = pd.DataFrame(bdata.obsm['X_umap'], index=bdata.obs_names)
umap_df = pd.concat([umap_df, bdata.obs.KO_gene], axis=1).reset_index()
umap_df.columns = ['gene','UMAP_x','UMAP_y','KO_gene']
px.scatter(umap_df, x='UMAP_x', y='UMAP_y', hover_name='gene', color='KO_gene', opacity=0.25)
# %%
umap_df.set_index('gene').loc[val_genes + test_genes, :]
# %%
#umap_df.iloc[g2i.loc[val_genes + test_genes + closest_known_gene[]], :]
# %%
sc.tl.tsne(bdata)
#sc.pl.tsne(bdata, color=['KO_gene'], groups=['KO_gene'], palette='Accent', legend_loc='right margin')
tsne_df = pd.DataFrame(bdata.obsm['X_tsne'], index=bdata.obs_names)
tsne_df = pd.concat([tsne_df, bdata.obs.KO_gene], axis=1).reset_index()
tsne_df.columns = ['gene','tSNE_x','tSNE_y','KO_gene']
px.scatter(tsne_df, x='tSNE_x', y='tSNE_y', hover_name='gene', color='KO_gene', opacity=0.25)

# %%
#bdata.X = bdata.obsm['X_pca']
reduce_scanpy(bdata, keep_scores=True, keep_loadings=False, keep_all_iters=False, layer=None, key_added='sca', iters=1, n_comps=50)
# %%
sca_df = pd.DataFrame(bdata.obsm['sca'], index=bdata.obs_names)
sca_df = pd.concat([sca_df, bdata.obs.KO_gene], axis=1).reset_index()
sca_df.columns = ['gene','SCA_x','SCA_y','KO_gene']
px.scatter(tsne_df, x='SCA_x', y='SCA_y', hover_name='gene', color='KO_gene', opacity=0.25)

# %%

plt.scatter(adata.X[], )