import pandas as pd
import numpy as np
import string

import os
import json

# Misc. utils.
def randstring():return "".join(np.random.choice(list(string.ascii_lowercase), (10,)))


# Make clusters 
class DataSet:
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
        self.X = pd.concat(out)
        
        
        self.MoA = np.concatenate([[i]*v for i,v in enumerate(self.size)])
    
        # Consistent names for columns and indices
        self.elementnames = [randstring() for i in range(len(self.X.index))]
        self.X.index = self.elementnames
        self.original_features = [randstring() for i in range(len(self.X.columns))] 
        

        
        self.X.columns = self.original_features
        self.MoA = pd.Series(self.MoA, index = self.elementnames)
        
                
        self.X = self.X.sample(frac=1) # re-order the datapoints soo that nothing 
                                     # can be accidentally inferred form their ordering.
            
        exec(self.transform_dataset) # apply a nonlinear transform to creat a new set od species
        
    

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

class DataSetSeries:

    def __init__(self,start_dataset, attr, attr_start, attr_range, n_trials = 6):
        
        self.start_dataset = start_dataset
        self.attr = attr
        self.attr_start = attr_start
        self.attr_range = attr_range
        self.n_trials = n_trials
        
        self.value_range = np.linspace(self.attr_start, 
                        self.attr_start + self.attr_range, 
                        self.n_trials
                       )
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

    def save(self, foldername = "scratch"):

        os.makedirs(foldername, exist_ok = True)
        dataset = self.start_dataset
        
        parameterdict = {'n_clusters':dataset.n_clusters,
        'dimension':dataset.dimension,
        'center_d':dataset.center_d,
        'scale':dataset.scale,
        'size':dataset.size,
        'attr':self.attr,
        'attr_start':self.attr_start,
        'attr_range':self.attr_range,
        'ellipticity':dataset.ellipticity,
        'size_range':dataset.size_range}

        fp = open(f"{foldername}/parameters.json",'w')
        json.dump(parameterdict, fp)
        fp.close()

        fp = open(f"{foldername}/transform_dataset.txt", 'w')
        fp.write(dataset.transform_dataset)
        fp.close()
    

# Sample to see inter-class distances
    
def weight2dist(u,v,d):
    weight = d['weight']
    return 1/weight - 1

class Sample:
    
    def __init__(self, dataset, frac = 0.4):
        list_ = [x.X[x.MoA == i].sample(frac = frac) for i in x.MoA.unique()]
        samples = pd.concat(list_)
        
        self.sampled_points = samples
        
        samples_per_class = [len(i) for i in list_]
        intra_class = scipy.linalg.block_diag(*[np.ones((k,k)) for k in samples_per_class])
        self.intra_class = intra_class + np.eye(sum(samples_per_class)) # 2 along diagonal i.e. i==j same point, 1 for i,j in same class, 0 if i,j in different classes
        
    def pairwise_network_distances(self):
        out = {}
        for i,v in nx.all_pairs_dijkstra_path_length(G, weight= weight2dist):
            out[i] = [v[i] for i in samples.index]
        distances = pd.DataFrame(out).transpose()
        distances.columns = samples.index
        distances = distances.loc[samples.index]
        return distances.values
    
    def spatial_distances(self, metric='euclidean'):
        return scipy.spatial.distance.cdist(samples, samples, metric = metric)
        
        
    def view_intra_versus_inter_distances(self, distances, ax = None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
        ax.hist(distances[intra_class == 1].flatten(), density = True);
        ax.hist(distances[intra_class == 0].flatten(), density = True);
        

        

class ClusteringMethodType:
    
    def __init__(self, name, color):
        self.name = name
        self.color = color

class ClusteringMethod:
    
    def __init__(self, methodtype, name_specific, function):
        self.methodtype = methodtype
        self.name_specific = name_specific
        self.function = function

    def cluster(self, dataset, scoring_method = sklearn.metrics.adjusted_rand_score):
        start_time = time.time()
        self.labels = self.function(dataset.X)
        end_time = time.time()
        evaluation_time = end_time - start_time
        self.labels = self.labels.reindex(dataset.X.index)
        score = scoring_method(self.labels, dataset.MoA)
        return score, evaluation_time
        
    def cluster_series(self, dataset_series):
        score_series = []
        time_series = []
        for dataset in dataset_series.datasets:
            #try:
            if True:
                score, evaluation_time = self.cluster(dataset)
                score_series.append(score)
                time_series.append(evaluation_time)
            if False:
                
            #except Exception:
                score_series.append(np.nan)
                time_series.append(np.nan)

        
        score_series = pd.Series(score_series, index = dataset_series.value_range)
        score_series.index.name = attr_description[dataset_series.attr]

        time_series = pd.Series(time_series, index = dataset_series.value_range)
        time_series.index.name = attr_description[dataset_series.attr] 
        
        return score_series, time_series

def evaluate(clustering_methods, dataset_series):
    out = {}
    out_time = {}
    
    # Gets the result of each clustering method on each individual dataset withing the series
    # to make a full table showing how each clustering method's performance decays
    for clustering_method in clustering_methods:
        temp_score,temp_time  = clustering_method.cluster_series(dataset_series)
        out[clustering_method.name_specific] = temp_score
        out_time[clustering_method.name_specific] = temp_time
        
    score_df = pd.concat(out, axis = 1)# index = dataset_series.value_range)
    time_df = pd.concat(out_time, axis= 1)# index = dataset_series.value_range)
    
    return score_df, time_df