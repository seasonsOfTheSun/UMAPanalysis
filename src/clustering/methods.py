import sklearn.metrics
import networkx as nx
import time
import numpy
import pandas

cmap_outer = {"black":"#000000",
"princeton-orange": "#ee8434ff",
"cobalt-blue": "#1446a0ff",
"razzmatazz": "#db3069ff",
"maximum-green": "#698f3fff",
"medium-purple": "#a682ffff",
"turquoise": "#42d9c8ff",
"mindaro": "#ddfc74ff",
"cyan-process": "#01baefff", 
"dark-pastel-green": "#20bf55ff", 
"orchid-pink": "#f6c0d0ff"}


""" Helper dataset tto wrap a pair of dataframes contain
    ing the dataponts and the labels from the datasets,
     allowing them to be quickly saved and loaded."""  


class RealDataSet:
    def __init__(self, data, labels, network, network_evaluation_time):
        self.data   = data
        self.labels = labels
        self.network = network
        self.network_evaluation_time = network_evaluation_time

def load_from_file(datafile, labelfile, networkfile, networkfile_et, column = "MeSH"):
    data  = pandas.read_csv(datafile,   index_col = 0)
    labelfile = pandas.read_csv(labelfile, index_col = 0)
    assert (data.index == labelfile.index).all()
    labels = labelfile[column]
    network = nx.read_gml(networkfile)
    
    fp = open(networkfile_et)
    network_evaluation_time = float(fp.read().strip())
    fp.close()
    
    return RealDataSet(data, labels, network, network_evaluation_time)

        
class ClusteringMethodType:
    
    def __init__(self, name, color, network_method = False):
        self.name = name
        self.color = color
        self.network_method = network_method

class ClusteringMethod:
    
    def __init__(self, methodtype, name_specific, function):
        self.methodtype = methodtype
        self.name_specific = name_specific
        self.function = function


    def cluster_data(self, data):
        
        start_time = time.time()
        self.labels = self.function(data) 
        end_time = time.time()
        self.evaluation_time = end_time - start_time        


    def cluster_network(self, network):
        
        start_time = time.time()
        self.labels = self.function(network)
        end_time = time.time()
        self.evaluation_time = end_time - start_time 

            
    def cluster(self, dataset, scoring_method = sklearn.metrics.adjusted_rand_score):

        if self.methodtype.network_method:
            self.cluster_network(dataset.network)
            self.evaluation_time += dataset.network_evaluation_time
        else:
            self.cluster_data(dataset.data)


        self.labels = self.labels.reindex(dataset.labels.index)
        masked = self.labels.isna() | dataset.labels.isna()
        score = scoring_method(self.labels[~masked], dataset.labels[~masked])
        
        return score, self.evaluation_time
        
    def cluster_series(self, dataset_series):
        score_series = []
        time_series = []
        for dataset in dataset_series.datasets:

            if True:
                score, evaluation_time = self.cluster(dataset)
                score_series.append(score)
                time_series.append(evaluation_time)
            if False:
                score_series.append(numpy.nan)
                time_series.append(numpy.nan)

        
        score_series = pandas.Series(score_series, index = dataset_series.value_range)
        score_series.index.name = dataset_series.attr

        time_series = pandas.Series(time_series, index = dataset_series.value_range)
        time_series.index.name = dataset_series.attr 
        
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
        
    score_df = pandas.concat(out, axis = 1)# index = dataset_series.value_range)
    time_df = pandas.concat(out_time, axis = 1)# index = dataset_series.value_range)
    
    return score_df, time_df


import sklearn.cluster

# k-Means clustering for baseline comparison
def kmeans(X, i):
    km = sklearn.cluster.KMeans(n_clusters = i)
    km.fit(X)
    return pandas.Series(km.labels_, index = X.index)

kmeans_type = ClusteringMethodType("k-Means", "#000000")


# OPTICS (Ordering Points To Infer Cluster Structure)
def optics(X):
    model = sklearn.cluster.OPTICS(min_samples=10, eps = 1000)
    clusters = model.fit_predict(X)
    return pandas.Series(clusters, index = X.index)
    
optics_type = ClusteringMethodType("Optics",  cmap_outer["turquoise"])


# Spectral Clustering
def spectral_cluster(X, i):
    model = sklearn.cluster.SpectralClustering(n_clusters=i)
    clusters = model.fit_predict(X)
    return pandas.Series(clusters, index = X.index)

sc_type = ClusteringMethodType("Spectral Clustering",  cmap_outer["turquoise"])


    
# Data-Net approach
import networkx as nx
import umap
import random

def umap_network(X):
    rndstate = numpy.random.RandomState(10)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, 'manhattan')
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

def greedyModularity(G):
    nodes = G.nodes()
    clusters = nx.community.modularity_max.greedy_modularity_communities(G)
    df = pandas.DataFrame([[i in a for a in clusters] for i in  nodes])
    df.index = nodes
    return df.idxmax(axis = 1)
        
gmtype = ClusteringMethodType('GreedyModularity', cmap_outer["medium-purple"], network_method = True)



import community.community_louvain
def louvain(G):
    return pandas.Series(community.community_louvain.best_partition(G))

lvtype = ClusteringMethodType('Louvain', cmap_outer["razzmatazz"], network_method = True)


# Autoencoder
import keras
from keras import layers

def autoencode(df, encoding_dim = 2, validation_split = 0.1):
    n = len(df.columns)
    df = (df - df.min()) / (df.max() - df.min())
    # This is the size of our encoded representations
    # This is our input image
    input_img = keras.Input(shape=(n,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(n, activation='sigmoid')(encoded)
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    ####|   As well as the decoder model   |####

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    #Now let's train our autoencoder to reconstruct MNIST digits.
    #First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adam optimizer:
    
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(df.values, df.values,
                    epochs=1000,
                    batch_size=256,
                    shuffle=True,
                    verbose=0,
                    validation_split = validation_split)
    
    codes = encoder.predict(df)
    return codes


import sklearn.cluster
def autencoded_clustering(df, encoding_dim = 2, validation_split = 0.0):
    codes = autoencode(df,encoding_dim=encoding_dim, validation_split =validation_split)
    km = sklearn.cluster.KMeans(n_clusters =10)
    km.fit(codes)
    return  pandas.Series(km.labels_, index = df.index)

autoencode_type = ClusteringMethodType("Autoencode",  cmap_outer["princeton-orange"])

    

    
clustering_methods = []

for i in range(2,3):
    clustering_methods.append(ClusteringMethod(autoencode_type, 
                                               f"{i}-Dimensional Autencoder",
                                                lambda X:autencoded_clustering(X, encoding_dim = i))
                             )

for i in range(8,10):
    clustering_methods.append(ClusteringMethod(sc_type, 
                                               f"Spectral Clustering {i} Dimensions",
                                                lambda X:spectral_cluster(X, i))
                             )

for i in range(1,20):
    clustering_methods.append(ClusteringMethod(kmeans_type,
                                               f"k-Means {i}",
                                                lambda X:kmeans(X, i))
                             )

clustering_methods.append(ClusteringMethod(optics_type, 
                                           "Optics",
                                            optics)
                         )

clustering_methods.append(ClusteringMethod(lvtype,
                                           'Louvain',
                                           louvain)
                         )

clustering_methods.append(ClusteringMethod(gmtype,
                                            'GreedyModularity',
                                            greedyModularity
                                            )
                         )

clustering_method_dict = {c.name_specific:c for c in clustering_methods}


