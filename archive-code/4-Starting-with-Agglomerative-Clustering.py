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
name="xclara.arff"

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


#####################################
#Methode du coude 
inerties_list = []
k_list = []
for k in range (1,50):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
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
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
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
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
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
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
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
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
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



#######################################################################