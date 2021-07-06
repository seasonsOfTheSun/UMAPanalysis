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




# k-Means clustering for baseline comparison
def kmeans(X, i):
    km = sklearn.cluster.KMeans(n_clusters = i)
    km.fit(X)
    return pd.Series(km.labels_, index = X.index)

kmeans_type = ClusteringMethodType("k-Means", "#000000")


# OPTICS (Ordering Points To Infer Cluster Structure)
def optics(X):
    model = sklearn.cluster.OPTICS(min_samples=10, eps = 1000)
    clusters = model.fit_predict(X)
    return pd.Series(clusters, index = X.index)
    
optics_type = ClusteringMethodType("Optics",  cmap_outer["turquoise"])


# Spectral Clustering
def spectral_cluster(X, i)
    model = sklearn.cluster.SpectralClustering(n_clusters=i)
    clusters = model.fit_predict(X)
    return pd.Series(clusters, index = X.index)

sc_type = ClusteringMethodType("Spectral Clustering",  cmap_outer["turquoise"])


    
# Data-Net approach
import networkx as nx
import umap
import random

def umap_network(X):
    rndstate = np.random.RandomState(10)
    nodes = list(X.index)
    G,_,_ = umap.umap_.fuzzy_simplicial_set(X/X.std(), 10, rndstate, 'manhattan')
    G = nx.from_scipy_sparse_matrix(G)
    return nx.relabel_nodes(G, dict(enumerate(X.index)).get)

def greedyModularity(X):
    G = umap_network(X)
    nodes = G.nodes()
    clusters = nx.community.modularity_max.greedy_modularity_communities(G)
    df = pd.DataFrame([[i in a for a in clusters] for i in  nodes])
    df.index = nodes
    return df.idxmax(axis = 1)
        
gmtype = ClusteringMethodType('GreedyModularity', cmap_outer["medium-purple"])



import community.community_louvain
def louvain(X):
    G = umap_network(X)
    return pd.Series(community.community_louvain.best_partition(G))

lvtype = ClusteringMethodType('Louvain', cmap_outer[""])


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

import matplotlib.pyplot as plt
import sklearn.cluster
def autencoded_clustering(df, encoding_dim = 2, validation_split = 0.0):
    codes = autoencode(df,encoding_dim=encoding_dim, validation_split =validation_split)
    km = sklearn.cluster.KMeans(n_clusters =10)
    km.fit(codes)
    return  pd.Series(km.labels_, index = df.index)

autoencode_type = ClusteringMethodType("Autoencode",  cmap_outer["princeton-orange"])

    

    
clustering_methods = []

for i in range(2,10):
    clustering_methods.append(ClusteringMethod(autoencode_type, 
                                               f"{i}-Dimensional Autencoder",
                                                lambda X:autencoded_clustering(X, encoding_dim = i)))

for i in range(1,10):
    clustering_methods.append(ClusteringMethod(sc_type, 
                                               f"Spectral Clustering {i} Dimensions",
                                                lambda X:spectral_cluster(X, i))
                             )
    
for i in range(1,20):
    clustering_methods.append(ClusteringMethod(kmeans_type, 
                                               "k-Means {i}", 
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