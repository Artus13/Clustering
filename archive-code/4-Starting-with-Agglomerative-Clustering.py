import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kneed import KneeLocator
import pretty_errors


###################################################################
# Exemple : Agglomerative Clustering


path = '../artificial/'
name="rings.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()



### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


#######################################################################

def evaluate_clustering(datanp, max_clusters=10):
    results = []
    
    # Tester plusieurs nombres de clusters
    for k in range(2, max_clusters+1):
        start_time = time.time()
        
        # Modèle de clustering agglomératif
        model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
        labels = model.fit_predict(datanp)
        
        # Temps de calcul
        runtime = time.time() - start_time
        
        # Calcul des métriques
        silhouette = silhouette_score(datanp, labels)
        davies_bouldin = davies_bouldin_score(datanp, labels)
        calinski_harabasz = calinski_harabasz_score(datanp, labels)
        
        # Stocker les résultats
        results.append({
            'k': k,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'runtime': runtime
        })
        
        # Affichage du clustering
        plt.scatter(datanp[:, 0], datanp[:, 1], c=labels, s=8)
        plt.title(f"Clustering agglomératif avec k={k}")
        plt.show()
    
    # Choisir le meilleur nombre de clusters en fonction de l'indice de silhouette (par exemple)
    best_k = max(results, key=lambda x: x['silhouette'])['k']
    print(f"Nombre de clusters optimal selon l'indice de silhouette: {best_k}")

    # Affichage des résultats pour chaque k
    for result in results:
        print(f"Clusters: {result['k']}, Silhouette: {result['silhouette']:.4f}, "
              f"Davies-Bouldin: {result['davies_bouldin']:.4f}, "
              f"Calinski-Harabasz: {result['calinski_harabasz']:.4f}, "
              f"Runtime: {result['runtime']:.4f} sec")

# Appel de la fonction pour évaluer le clustering
evaluate_clustering(datanp, max_clusters=10)