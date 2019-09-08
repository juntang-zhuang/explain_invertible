import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from torch.utils.data.dataset import Dataset
import torch
from sklearn.datasets.samples_generator import make_blobs
# ----------------------------------------------------------------------------------------
# 2-class example
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1,
#                           n_samples= 1000)
X, y = make_moons(noise=0.2, random_state=0, n_samples=1000)
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=1000)
num_classes = 2
# --------------------------------------------------------------------------------------
# 3-class example
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, y = make_blobs(n_samples=10000, centers=centers, cluster_std=0.4,random_state=0)
#num_classes = 3
# --------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
class Data1D(Dataset):
    def __init__(self,split='train'):
        super(Data1D,self).__init__()
        self.split = split
        if self.split == 'train':
            self.x = X_train
            self.y = y_train
        elif self.split == 'test':
            self.x = X_test
            self.y = y_test

    def __getitem__(self, item):
        return torch.from_numpy(self.x[item]).float(), torch.from_numpy(np.array(self.y[item])).long()

    def __len__(self):
        return self.x.shape[0]
