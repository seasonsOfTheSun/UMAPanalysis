
#dataset="cytodata"
#dataset="lish-moa"
dataset="toy"
for i in $(seq 0 1 20)
do
    python src/network.py ${dataset} $i manhattan 
    #python src/network.py ${dataset} euclidean $i
done


# Visualization
#for i in $(ls -d networks/${dataset}/*.gml)
#do
#    python src/visualize/sparsify.py $i
#done

#for i in $(ls -d networks/${dataset}/sparsified/*.gml)
#do
#        python src/visualize/layout.py $i
#done

#for i in $(ls -d networks/${dataset}/coarsened/*.gml)
#do
#        python src/visualize/layout.py $i
#done

# Clustering
for i in $(seq 0 1 20)
do
    for j in $(seq 0 10 200)
    do
        python src/clustering/make_noisy_networks.py $i ${dataset} manhattan $j
#        python src/clustering/make_noisy_networks.py ${dataset} $i euclidean $j
    done
done

for i in networks/${dataset}/noisy/*.gml
do
#    python src/clustering/fiedler.py $i
    python src/clustering/louvain.py $i
done

for j in $(seq 0 10 200)
do
    python src/clustering/kmeans.py ${dataset} $j
    python src/clustering/spectral.py ${dataset} $j
done

for i in $(ls -d data/processed/clusters/${dataset}/*.csv)
do
    python src/clustering/enrichment.py $i
done

# Classification
for i in networks/${dataset}/*.gml
do
    echo $i
    python src/classification/predict_with_nearest_neighbors.py $i
    python src/classification/predict_with_umap.py $i
done

for moa in data/intermediate/${dataset}/classes/*
do
    python src/classification/predict_with_random_forest.py $dataset $moa
done

for i in data/processed/predictions/${dataset}/*.csv
do
    python src/classification/draw_roc_curve.py $i
done

# Centrality
for i in networks/${dataset}/*.gml
do
    python src/centrality/eigenvector_centrality.py $i
done
