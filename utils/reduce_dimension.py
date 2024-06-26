from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def reduce_dimension(input, output_n, reduce_type='pca', **kargs):
    print(f'Start dimensionality reduction, dimensionality reduction method {reduce_type}, Input dimension {input.shape[0]}, Output dimension {output_n}')
    assert isinstance(input, np.ndarray)
    if input.shape[0] < input.shape[1]:
        print(f'WARNING, Input vector dimension{input.shape}, The input dimension should be(samples, n_features)')

    if reduce_type == 'tsne':
        output = TSNE(n_components=output_n, init='pca').fit_transform(input)
    elif reduce_type == 'pca':
        output = PCA(n_components=output_n).fit_transform(input)
    elif reduce_type == 'pca_tsne':
        pca = PCA(n_components=0.99)
        x = pca.fit_transform(input)
        print(f'pca noise_variance {pca.noise_variance_}')
        tsne = TSNE(n_components=output_n, init='pca')
        output = tsne.fit_transform(x)
    else:
        raise Exception('ERROR, reduce_type must in tsne/pca/pca_tsne')
    print(f'Dimension reduction completed')
    return output


if __name__ == "__main__":

    reduce_dimension(np.array([[1,2],[2,3],[3,4]]), 2)


