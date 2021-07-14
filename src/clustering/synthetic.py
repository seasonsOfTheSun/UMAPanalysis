import pandas as pd
import numpy as np
import string
import time

import os
import json

# Misc. utils.
def randstring():return "".join(np.random.choice(list(string.ascii_lowercase), (10,)))

import networkx as nx
import umap
import random
def umap_network(X):
    rndstate = np.random.RandomState(10)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, 'manhattan')
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)


# Make clusters 
class SyntheticDataSet:
    """ """
    
    def __init__(self, n_clusters, dimension, center_d, scale, size, ellipticity = 0, scale_range=0, center_d_range=0, size_range=0, transform_dataset = "pass"):
        self.n_clusters = n_clusters
        self.dimension = dimension
        self.scale = scale
        self.center_d = center_d
        self.size = size
        self.scale_range=scale_range
        self.center_d_range=center_d_range
        self.size_range=size_range
        self.ellipticity=ellipticity
        self.transform_dataset = transform_dataset
        

    def vary_clusters(self):
        """
        Make the variables controlling the clusters (e.g. cluster size) 
        vary according to some predefined range.
        
        """
        n_clusters = self.n_clusters
        
        if type(self.scale) != list:
            self.scale = [self.scale+(i-0.5*n_clusters)*self.scale_range/n_clusters for i in range(n_clusters)]

        if type(self.center_d) != list:
            self.center_d = [self.center_d+(i-0.5*n_clusters)*self.center_d_range/n_clusters for i in range(n_clusters)]
        
        if type(self.size) != list:
            self.size = [int(self.size+(i-0.5*n_clusters)*self.size_range/n_clusters) for i in range(n_clusters)]
    
    def make_dataset(self):
        """ Create the features from the parameters given, then """
        
        self.vary_clusters()
        
        # Make cluster centers
        randdirs = np.random.randn(self.n_clusters, self.dimension)
        randdirs = randdirs / np.sqrt((randdirs**2).sum(axis = 1)).reshape((self.n_clusters,1))
        self.centers = np.array([[i] for i in self.center_d]) * randdirs
        
        out = []# Make the data points within each cluster and scale and distribute accordingly
        for i,n in enumerate(self.size):
            temp = self.scale[i]*np.random.randn(n, self.dimension)
            
            # Add variation to the clusters along different axes so they are less spherical
            temp = (1 + self.ellipticity*np.random.rand(self.dimension)).reshape(-1,self.dimension) * temp
            temp = pd.DataFrame(temp)
            temp = temp + self.centers[i,:]
            out.append(temp)



        # Join into dataframes
        self.data= pd.concat(out)
        self.labels = np.concatenate([[i]*v for i,v in enumerate(self.size)])
    
        # Consistent names for columns and indices
        self.elementnames = [randstring() for i in range(len(self.data.index))]
        self.data.index = self.elementnames
        self.original_features = [randstring() for i in range(len(self.data.columns))]


        self.data.columns = self.original_features
        self.labels = pd.Series(self.labels, index = self.elementnames)
        
                
        self.data = self.data.sample(frac=1) # re-order the datapoints soo that nothing 
                                     # can be accidentally inferred form their ordering.
            
        exec(self.transform_dataset) # apply a nonlinear transform to creat a new set of features
        
        start_time = time.time()
        self.network = umap_network(self.data)
        end_time = time.time()
        self.network_evaluation_time = end_time - start_time


# Add nonlinear features
def bubble_value(df, radius):
    S_2 = (df**2).sum(axis = 1)
    bump = radius**2 - S_2
    return bump.map(lambda x: np.sqrt(x) if x >= 0 else 0)

def sine_mapping_value(df, n, amplitude, period):
    return amplitude * np.sin(df[n]/period)



# Make series of datasets with one variable changed
import copy

attr_description = {'n_clusters':'Number of Clusters', 'dimension':'Number of Features',
'scale':'Cluster Radius',
 'center_d':'Seperation of Clusters', 
 'size':'Size of Clusters',
'scale_range':'Variation in Cluster Radius', 
 'center_d_range':'Variation in Seperation of Clusters', 
 'size_range':'Variation in Size of Clusters',
 'ellipticity':'Deviation from Sphericity of the CLuster Shapes'}



series_type = {'n_clusters':'mean', 'dimension':'mean',
'scale':'mean', 'center_d':'mean', 'size':'mean','ellipticity':'mean',
'scale_range':'var', 'center_d_range':'var', 'size_range':'var'}

typeof_parameter = {'n_clusters':int, 'dimension':int,
'scale':float, 'center_d':float, 'size':int,'ellipticity':float,
'scale_range':float, 'center_d_range':float, 'size_range':int}

class SyntheticDataSetSeries:

    def __init__(self,start_dataset, attr, value_range):
        
        self.start_dataset = start_dataset
        self.attr = attr
        
        self.value_range = value_range

        type_ = typeof_parameter[self.attr]
        self.value_range = [type_(i) for i in self.value_range]
    
    def datasetMeanSeries(self):
        """
        Make the series by changin the value of one of the parameters, 
        or the mean value if the parameter varies by cluster within the dataset.
        """
        out = []
        for i in self.value_range:
            copied_dataset = copy.deepcopy(self.start_dataset)
            copied_dataset.__setattr__(self.attr, i)
            copied_dataset.vary_clusters()
            copied_dataset.make_dataset()
            out.append(copied_dataset)
        self.datasets = out

    def datasetVarianceSeries(self):
        """
        Make the series by change in the variance of one of 
        the parameters across the clusters within the dataset.
        """
        
        out = []
        for i in self.value_range:
            copied_dataset = copy.deepcopy(self.start_dataset)
            copied_dataset.vary_clusters(**{self.attr:i})
            copied_dataset.make_dataset()
            out.append(copied_dataset)
        self.datasets = out

    def make_series(self):
        seriesmaker = series_type[self.attr]
        if seriesmaker == 'mean':
            self.datasetMeanSeries()
        elif seriesmaker == 'var':
            self.datasetVarianceSeries()

    def save(self, foldername = "scratch",  save_tabular_data = True):

        os.makedirs(foldername, exist_ok = True)
        dataset = self.start_dataset

        parameterdict = {'n_clusters':dataset.n_clusters,
        'dimension':dataset.dimension,
        'center_d':dataset.center_d,
        'scale':dataset.scale,
        'size':dataset.size,
        'attr':self.attr,
        'value_range':list(self.value_range),
        'ellipticity':dataset.ellipticity,
        'size_range':dataset.size_range,
        'scale_range':dataset.scale_range,
        'center_d_range':dataset.center_d_range,
        'size_range':dataset.size_range}

        fp = open(f"{foldername}/parameters.json",'w')
        json.dump(parameterdict, fp)
        fp.close()

        fp = open(f"{foldername}/transform_dataset.txt", 'w')
        fp.write(dataset.transform_dataset)
        fp.close()
        
        if save_tabular_data == True:
            for i, dataset in enumerate(self.datasets):
                os.makedirs(f"{foldername}/dataset_{i}", exist_ok = True)
                dataset.data.to_csv(f"{foldername}/dataset_{i}/features.csv")
                dataset.labels.to_csv(f"{foldername}/dataset_{i}/labels.csv")

def load(foldername):

    parameterdict = json.load(open(f"{foldername}/parameters.json"))

    fp = open(f"{foldername}/transform_dataset.txt")
    transform = fp.read()

    start_dataset = SyntheticDataSet(parameterdict['n_clusters'],
                                     parameterdict['dimension'],
                                     parameterdict['center_d'],
                                     parameterdict['scale'],
                                     parameterdict['size'],
                                     parameterdict['ellipticity'],
                                     parameterdict['scale_range'],
                                     parameterdict['center_d_range'],
                                     parameterdict['size_range'],
                                     transform_dataset = transform)

    
    n_trials = parameterdict['n_trials']

    dataset_series = SyntheticDataSetSeries(start_dataset,parameterdict['attr'],np.array(parameterdict['value_range']))
    
    dataset_series.make_series()
    
    for i in range(n_trials):
        dataset_series.datasets[i].data = pd.read_csv(f"{foldername}/dataset_{i}/features.csv", index_col = 0)
        dataset_series.datasets[i].labels = pd.read_csv(f"{foldername}/dataset_{i}/labels.csv", index_col = 0)
    return dataset_series

# Sample to see inter-class distances    
def weight2dist(u,v,d):
    weight = d['weight']
    return 1/weight - 1

if False:

    n_clusters = 10
    dimension = 10
    center_d = 1
    scale = 0.1
    size = 30
    ellipticity = 5
    size_range = 0
    
    attr = 'scale'
    attr_start = 0.1
    attr_range = 2.1
    n_trials = 10
    value_range = np.linspace(attr_start, attr_range, n_trials)

    dataset =     SyntheticDataSet(n_clusters,
                                   dimension, 
                                   center_d,
                                   scale,
                                   size,
                                   ellipticity = ellipticity, 
                                   size_range=size_range)

    dataset_series = SyntheticDataSetSeries(dataset,
                                            attr,
                                            value_range)

    
    dataset_series.make_values()


