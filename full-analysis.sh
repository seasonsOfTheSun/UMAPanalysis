
dataset="cytodata"
dataset="lish-moa"
for i in $(seq 0 1 20)
do
    ./pythonb src/network.py ${dataset} manhattan $i
    ./pythonb src/network.py ${dataset} euclidean $i
done


# Visualization
for i in $(ls -d networks/${dataset}/*.gml)
do
    ./pythonb src/visualize/sparsify.py $i
done

for i in $(ls -d networks/${dataset}/sparsified/*.gml)
do
        ./pythonb src/visualize/layout.py $i
done

for i in $(ls -d networks/${dataset}/coarsened/*.gml)
do
        ./pythonb src/visualize/layout.py $i
done

# Clustering
for i in $(seq 0 1 20)
do
    for j in $(seq 0 10 200)
    do
        ./pythonb src/clustering/make_noisy_networks.py ${dataset} $i manhattan $j
        ./pythonb src/clustering/make_noisy_networks.py ${dataset} $i euclidean $j
    done
done

for i in networks/${dataset}/noisy/*.gml
do
#    ./pythonb src/clustering/fiedler.py $i
    ./pythonb src/clustering/louvain.py $i
done

for j in $(seq 0 10 200)
do
#    ./pythonb src/clustering/kmeans.py ${dataset} $j
    ./pythonb src/clustering/spectral.py ${dataset} $j
done

for i in $(ls -d data/processed/clusters/${dataset}/*.csv)
do
    ./pythonb src/clustering/enrichment.py $i
done

# Classification
for i in networks/${dataset}/*.gml
do
    echo $i
    ./pythonb src/classification/predict_with_nearest_neighbors.py $i
    ./pythonb src/classification/predict_with_umap.py $i
done

for moa in data/intermediate/${dataset}/classes/*
do
    ./pythonb src/classification/predict_with_random_forest.py $dataset $moa
done

for i in data/processed/predictions/${dataset}/*.csv
do
    ./pythonb src/classification/draw_roc_curve.py $i
done

# Centrality
for i in networks/${dataset}/*.gml
do
    ./pythonb src/centrality/eigenvector_centrality.py $i
done
