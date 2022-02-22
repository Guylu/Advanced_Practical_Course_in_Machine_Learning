import densmap
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
import matplotlib.pyplot as plt

digits = fetch_openml(name='mnist_784')
subsample, subsample_labels = resample(digits.data, digits.target, n_samples=7000,
                                       stratify=digits.target, random_state=1)

embedding, ro, re = densmap.densMAP().fit_transform(subsample)

plt.scatter(embedding)
plt.show()