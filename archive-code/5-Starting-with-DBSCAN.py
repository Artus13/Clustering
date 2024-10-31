"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

##################################################################
# Chargement des données

path = '../artificial/'
name = "banana.arff"
databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# Visualisation des données initiales
print("Affichage des données initiales : " + str(name))
f0 = datanp[:, 0]  # Première colonne
f1 = datanp[:, 1]  # Deuxième colonne
plt.scatter(f0, f1, s=8)
plt.title("Données initiales : " + str(name))
plt.show()

##################################################################
# Standardisation des données
scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
f0_scaled = data_scaled[:, 0]
f1_scaled = data_scaled[:, 1]
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Données standardisées")
plt.show()

##################################################################
# 5.2 Méthode du coude pour déterminer le paramètre eps

k = 5  # Nombre de voisins
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(data_scaled)
distances, _ = neigh.kneighbors(data_scaled)
# Calcul des distances moyennes aux k plus proches voisins
newDistances = np.asarray([np.mean(distances[i][1:]) for i in range(distances.shape[0])])
distancetrie = np.sort(newDistances)

# Affichage de la courbe des k-plus proches voisins
plt.title(f"Méthode du coude pour k={k} plus proches voisins")
plt.plot(distancetrie)
plt.xlabel("Points")
plt.ylabel("Distance moyenne aux k plus proches voisins")
plt.show()

##################################################################
# 5.3 Exécution du clustering DBSCAN avec évaluation des métriques

# Paramètres DBSCAN
epsilon = 0.05  
min_samples = 5

# Clustering DBSCAN
print("Appel DBSCAN avec eps = ", epsilon, "et min_samples = ", min_samples)
start_time = time.time()
model = cluster.DBSCAN(eps=epsilon, min_samples=min_samples)
model.fit(data_scaled)
run_time = time.time() - start_time

labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

# Calcul des métriques 
if n_clusters > 1:
    silhouette_avg = silhouette_score(data_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
    davies_bouldin = davies_bouldin_score(data_scaled, labels)

    print(f"Indice de silhouette : {silhouette_avg:.4f}")
    print(f"Indice de Calinski-Harabasz : {calinski_harabasz:.4f}")
    print(f"Indice de Davies-Bouldin : {davies_bouldin:.4f}")

# Affichage des résultats
print(f"Nombre de clusters : {n_clusters}")
print(f"Nombre de points de bruit : {n_noise}")
print(f"Temps d'exécution : {run_time:.4f} secondes")

# Visualisation des clusters
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=8, cmap='viridis')
plt.title(f"Clustering DBSCAN - eps={epsilon}, MinPts={min_samples}")
plt.show()
