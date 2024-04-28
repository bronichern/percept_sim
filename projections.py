
import umap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

def umap(
    data_subset,
    n_comp=3,
    n_neighbors=50,
    min_dist=0.1,
    init="spectral",
    random_state=123123,
    densmap=False,
    dens_lambda=2.0,
    dens_frac=0.3,
):
    """
    n_neighbors: This determines the number of neighboring points used in local approximations of manifold structure. Larger values will result in more global structure being preserved at the loss of detailed local structure. In general this parameter should often be in the range 5 to 50, with a choice of 10 to 15 being a sensible default.
    min_dist: This controls how tightly the embedding is allowed compress points together. Larger values ensure embedded points are more evenly distributed, while smaller values allow the algorithm to optimise more accurately with regard to local structure. Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default.
    metric: This determines the choice of metric used to measure distance in the input space. A wide variety of metrics are already coded, and a user defined function can be passed as long as it has been JITd by numba.
    """

    return umap.UMAP(
        random_state=random_state,
        min_dist=min_dist,
        init=init,
        densmap=densmap,
        dens_lambda=dens_lambda,
        n_components=n_comp,
        n_neighbors=n_neighbors,
    ).fit_transform(data_subset)

def kpca(data_subset, n_comp=3, kernel="rbf", gamma=None, random_state=123123, degree=3):
    normalized_ds = MinMaxScaler().fit_transform(data_subset)
    pca = KernelPCA(n_components=n_comp, kernel=kernel, gamma=gamma, random_state=random_state, degree=degree)
    pca_results = pca.fit_transform(normalized_ds)
    return pca_results

def tsne(data_subset, init="random", early_exaggeration=12.0, lr=200.0, n_comp=2, perplexity=40, iters=300, seed=65854):
    tsne = TSNE(
        n_components=n_comp,
        verbose=1,
        perplexity=perplexity,
        n_iter=iters,
        init=init,
        early_exaggeration=early_exaggeration,
        learning_rate=lr,
        random_state=seed,
    )
    
    tsne_results = tsne.fit_transform(data_subset)
    return tsne_results