dataset="cytodata"
for i in $(seq 0 1 20)
do
    ./pythonb src/networks.py $dataset $i
done

#visualization
./pythonb src/coarsen.py

#clustering
for i in $(seq 0 1 20)
do
    for i in $(seq 0 10 200)
    do
        ./pythonb src/clustering/make_noisy_networks.py $dataset $i
    done
done


for i in $(ls networks/${dataset}/noisy/eigenvectors/)
do
    ./pythonb src/clustering/fiedler.py networks/${dataset}/noisy/eigenvectors/$i
done

./pythonb src/clustering/kmeans.py $dataset

for i in $(ls data/processed/predictions/$dataset)
do
    ./pythonb src/clustering/enrichment.py data/processed/predictions/$dataset/$i
done




