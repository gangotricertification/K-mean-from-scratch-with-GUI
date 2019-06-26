import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog,Tk



fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')




#------------------------------Tkinter to create dialog for selecting the text file-------------------------------------


root = Tk()
root.fileName=filedialog.askopenfilename(filetypes = (("choose text file","*.txt"),("All files","*.*")))
print(root.fileName)
data = np.genfromtxt(root.fileName,delimiter=" ")
print(data)

X = data
x=X.T

colors = 10*["g","r","c","b","k"]   # array of different type of colors to be used depending on no of clusters

#--------------------------K_Means class to fully implement K mean from scratch----------------------------------------


class K_Means:
    def __init__(self, k_):
        self.k = k_
        self.tol = 0.001
        self.max_iter = 300   # defined no of iteration to used get satisfactory centroids of clusters

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):     # choose random point to intializing k mean
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []    # initally empty dictionary with no values only no of clusters

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]  # find the euclidean distance from each centroid
                classification = distances.index(min(distances))   # choose the centroid fromm where the distance is minimum to point
                self.classifications[classification].append(featureset)    # now append the point or feature set into the key according to cluster
            prev_centroids = dict(self.centroids)   # whole centoids dictionary in previous_centoids as dictionary


            for classification in self.classifications:   # here find the average of all the points collected in for each centroid and it form new centroid
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
        return self.centroids


#-------------------------to predict where the point lies in cluster-----------------------------------------------------


    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        points = classificaitons[classification]
        return points   #classification


#--------------------------code to enter number of cluster---------------------------------------------------------------


clusters=int(input("\nno of clusters or k\n"))
clf = K_Means(clusters)
cd = clf.fit(X)
print("\ncentroids of all clusters")
print(cd)

#--------------------to plot 3d graph for 3dimensional data-------------------------------------------------------------
if clusters==3:
    for classification in clf.classifications:
        color = colors[classification]
        q=clf.classifications
        B = np.array(list(q[classification]))
        b = B.T
        ax.scatter(b[0],b[1],b[2],marker="x", color=color, s=150, linewidths=5)
        ################
        A = np.array(cd[classification])
        a = A.T
        ax.scatter(a[0],a[1],a[2],color=color,linewidth=5,marker='o')
##    A = np.array(list(cl.values()))
##    a = A.T
##    ax.scatter(a[0],a[1],a[2],c='r',marker='o')
    print(cd)
    plt.show()

#--------------code to write the predicted centroids into the text file named "submission.txt"--------------------------


f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
if f is None:
    print("Save cancelled")
    
for i in cd:
    for j in cd[i]:
	    f.write("%s "%j)
    f.write("\n")
f.close()

print("\nfile is saved ")
print("\ngangotricertification@gmail.com \n")
