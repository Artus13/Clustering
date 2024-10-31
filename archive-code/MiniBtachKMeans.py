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
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kneed import KneeLocator
import pretty_errors

##################################################################
# Exemple :  k-Means Clustering

path = '../artificial/'
name="cluto-t8-8k.arff"

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

#Methode du coude 
inerties_list = []
k_list = []
for k in range (1,50):
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    inertie = model.inertia_
    inerties_list.append(inertie)
    k_list.append(k)
coude = KneeLocator(k_list, inerties_list, curve = "convex", direction = "decreasing").elbow
plt.plot(k_list, inerties_list, marker='o')
plt.axvline(x=coude, linestyle='--', color='red', label=f'Coude à k={coude}')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters k ")
plt.ylabel("Inertie")
plt.legend()
plt.show()
print(f'Coude à k={coude}')

#Methode silhouette
silhouette_list = []
k_list = []
for k in range (2,50):
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    # informations sur le clustering obtenu
    score = silhouette_score(datanp, labels)
    silhouette_list.append(score)
    k_list.append(k)
k_opt = k_list[np.argmax(silhouette_list)]
plt.plot(k_list, silhouette_list, marker='o')
plt.axvline(x=k_opt, linestyle='--', color='red', label=f'Optimal à k={k_opt}')
plt.title("Méthode Silhouette")
plt.xlabel("Nombre de clusters k ")
plt.ylabel("Score de silhouette")
plt.legend()
plt.show()
print(f'Optimal à k={k_opt}')

#Methode Davies-Bouldin
db_list = []
k_list = []
for k in range (2,50):
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    # informations sur le clustering obtenu
    score = davies_bouldin_score(datanp, labels)
    db_list.append(score)
    k_list.append(k)
k_opt = k_list[np.argmin(db_list)]
plt.plot(k_list, db_list, marker='o')
plt.axvline(x=k_opt, linestyle='--', color='red', label=f'Optimal à k={k_opt}')
plt.title("Méthode Davies-Bouldin")
plt.xlabel("Nombre de clusters k ")
plt.ylabel("Score de Davies-Bouldin")
plt.legend()
plt.show()
print(f'Optimal à k={k_opt}')

#Methode Calinski-Harabasz
ch_list = []
k_list = []
for k in range (2,50):
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    # informations sur le clustering obtenu
    score = calinski_harabasz_score(datanp, labels)
    ch_list.append(score)
    k_list.append(k)
k_opt = k_list[np.argmax(ch_list)]
plt.plot(k_list, ch_list, marker='o')
plt.axvline(x=k_opt, linestyle='--', color='red', label=f'Optimal à k={k_opt}')
plt.title("Méthode Calinski-Harabasz")
plt.xlabel("Nombre de clusters k ")
plt.ylabel("Score de Calinski-Harabasz")
plt.legend()
plt.show()
print(f'Optimal à k={k_opt}')

#Methode RunTime
runtime_list = []
k_list = []
for k in range (1,50):
    tps1 = time.time()
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    tps2 = time.time()
    # informations sur le clustering obtenu
    runtime = round((tps2 - tps1)*1000,2)
    runtime_list.append(runtime)
    k_list.append(k)
k_opt = k_list[np.argmin(runtime_list)]
plt.plot(k_list, runtime_list, marker='o')
plt.axvline(x=k_opt, linestyle='--', color='red', label=f'Optimal à k={k_opt}')
plt.title("Méthode RunTime")
plt.xlabel("Nombre de clusters k ")
plt.ylabel("RunTime")
plt.legend()
plt.show()
print(f'Optimal à k={k_opt}')


#Run clustering method for a given number of clusters
# print("-------------------------------------------------------------------------")
# print("Appel KMeans pour une valeur de k fixée")
# tps1 = time.time()
# k=3
# model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
# model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# # informations sur le clustering obtenu
# iteration = model.n_iter_
# inertie = model.inertia_
# centroids = model.cluster_centers_

# #plt.figure(figsize=(6, 6))
# plt.scatter(f0, f1, c=labels, s=8)
# plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
# plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
# #plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
# plt.show()

# print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

# from sklearn.metrics.pairwise import euclidean_distances
# dists = euclidean_distances(centroids)
# print(dists)

