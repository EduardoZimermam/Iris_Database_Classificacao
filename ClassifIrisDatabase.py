#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import tree
import graphviz

def plotDiagram2D(featA, featB, nameFeatA, nameFeatB):
	x_min, x_max = X[:, featA].min() - .5, X[:, featA].max() + .5
	y_min, y_max = X[:, featB].min() - .5, X[:, featB].max() + .5
	
	plt.figure('{} x {}'.format(nameFeatA, nameFeatB), figsize=(8, 6))
	plt.clf()
	
	# Plot the training points
	plt.scatter(X[:, featA], X[:, featB], c=y, cmap=plt.cm.Set1,
	            edgecolor='k')
	plt.xlabel(nameFeatA)
	plt.ylabel(nameFeatB)
	
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())

def plotDiagram3D(featA, featB, featC, nameFeatA, nameFeatB, nameFeatC):
	fig = plt.figure("{} x {} x {}".format(nameFeatA, nameFeatB, nameFeatC), figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	X_reduced = PCA(n_components=4).fit_transform(iris.data)
	ax.scatter(X_reduced[:, featA], X_reduced[:, featB], X_reduced[:, featC], c=y,
	           cmap=plt.cm.Set1, edgecolor='k', s=40)
	ax.set_title("{} x {} x {}".format(nameFeatA, nameFeatB, nameFeatC))
	ax.set_xlabel(nameFeatA)
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel(nameFeatB)
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel(nameFeatC)
	ax.w_zaxis.set_ticklabels([])

# Load do dataset
iris = datasets.load_iris()

# Extração de 4 características
X = iris.data[:, :4]
y = iris.target

# -------------- PARA PLOTAR OS GRÁFICOS 2D ----------------- #
# plotDiagram2D(0, 1, 'Sepal length', 'Sepal width')
# plotDiagram2D(0, 2, 'Sepal length', 'Petal length')
# plotDiagram2D(0, 3, 'Sepal length', 'Petal width')
# plotDiagram2D(1, 2, 'Sepal width', 'Petal length')
# plotDiagram2D(1, 3, 'Sepal width', 'Petal width')
# plotDiagram2D(2, 3, 'Petal length', 'Petal width')

# # -------------- PARA PLOTAR OS GRÁFICOS 3D ----------------- #
# plotDiagram3D(0, 1, 2, 'Sepal length', 'Sepal width', 'Petal length')
# plotDiagram3D(0, 1, 3, 'Sepal length', 'Sepal width', 'Petal width')
# plotDiagram3D(0, 2, 3, 'Sepal length', 'Petal length', 'Petal width')
# plotDiagram3D(1, 2, 3, 'Sepal width', 'Petal length', 'Petal width')

plt.show()

# ------------------ ÁRVORE DE DECISÃO GINI ---------------------- #

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("entropy") 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("iris") 
