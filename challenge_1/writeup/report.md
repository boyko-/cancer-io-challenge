## My approach to Challenge 1 is very simple:

### A) For each query gene (from the validation or test set), figure out which of the training genes are "close"
1) First transpose the log1+normalized expression matrix adata
2) Next, compute linear or non-linear embeddings of the genes using PCA, UMAP, tSNE, (SCA did not work due to memory constraints)
2') The embedding can also be performed on a pariwise distance matrix between the genes, but results should be similar
3) Finally, find K nearest neighbors of each query gene, using multiple embeddings. This step is semi-manual becuase I chose different values of K for each query gene depending on the distance to its neighbors   

### B) For each query gene, predict that knocking it out would result in a state distribution that is a simple function of the observed state distributions for its neighbors among the training genes
1) For each neighbor gene, sum the cells counts in each state across gRNAs targeting this neighbor gene
2) In past submissions, I normalized the state distribution to sum=1 for each neighbor gene, but I skip this step in the final submission because some genes have very small cell counts with more uncertainty in their state distributions
3) Then sum the cell counts across all neighbor genes before normalizing the state distribution to sum=1.

### That's all. I had some ideas but no time to implement fancy steps like imputing expression dropout or bootstrap resampling the cell counts.