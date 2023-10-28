from __future__ import absolute_import

import faiss
import os

import numpy as np


from neurips23.ood.base import BaseOODANN
from benchmark.datasets import DATASETS, download_accelerated

class WMood(BaseOODANN):
    def __init__(self, metric, index_params):
        self.name = "diskann"

        self.index_params = index_params
        print("the mtric given as input is {}".format(metric))
        self.metric = metric
        self.indexkey = index_params.get("indexkey")


    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"

    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return faiss.METRIC_L2
        elif metric == 'ip':
            return faiss.METRIC_INNER_PRODUCT
        else:
            raise Exception('Invalid metric')
        

    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        """

        ds = DATASETS[dataset]()
        print('the stored metric ', self.metric)
        metric = self.translate_dist_fn(self.metric)
        print("metric translated ", metric)
        index = faiss.index_factory(ds.d, self.indexkey)
        index.metric = metric
        xb = ds.get_dataset()

        queries = ds.get_queries()
        queries_shape = queries.shape
        print('the queries shape is ', queries_shape)

        train_data = np.vstack([queries, xb[:queries_shape[0]*2]])

        print("train shape: ", train_data.shape)
        index.train(train_data)
        print("populate")

        index.add(xb)

        self.index = index
        self.nb = ds.nb

        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        print("store", self.index_name(dataset))
        faiss.write_index(index, self.index_name(dataset))




    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
                return False

        print("Loading index")

        self.index = faiss.read_index(self.index_name(dataset))
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        return True
        
    def query(self, X, k):
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')
        bs = 1024
        for i0 in range(0, nq, bs):
            _, self.I[i0:i0+bs] = self.index.search(X[i0:i0+bs], k)

    def get_results(self):
        return self.I

    def set_query_arguments(self, query_args):
        faiss.cvar.indexIVF_stats.reset()
        if "nprobe" in query_args:
            self.nprobe = query_args['nprobe']
            self.ps.set_index_parameters(self.index, f"nprobe={query_args['nprobe']}")
            self.qas = query_args
        else:
            self.nprobe = 1


    def __str__(self):
        return f'WMood({self.indexkey, self.qas})'