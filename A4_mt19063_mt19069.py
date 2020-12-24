import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preprocessing(data):

  

  # print(data.columns)
  cols=['Elevation','Aspect','Slope','Wilderness','Soil_Type','Hillshade_9am','Hillshade_noon','Horizontal_Distance_To_Fire_Points']
  one_hot=pd.get_dummies(data.drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','id'],axis=1))
  # print(one_hot.columns)
  df=data.drop(cols,axis=1)
  df=df.join(one_hot)
  print(df.isna().any().any())
  df.drop(['id'],axis=1,inplace=True)
  # df.head(5)
  # print(df.info())
  return df

data=pd.read_csv("clustering_data.csv")
df=preprocessing(data)

def two_D_visualization(df,t,cl):
  print("2-D visualization:")
  np.random.seed(42)
  rndperm = np.random.permutation(df.shape[0])
  if(t):
    A="y"
  else:
    A=None
  pca = PCA(n_components=3)
  pca_result = pca.fit_transform(df.values)
  df['pca-one'] = pca_result[:,0]
  df['pca-two'] = pca_result[:,1] 
  df['pca-three'] = pca_result[:,2]
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="pca-one", y="pca-two",
      hue=A,
      palette=sns.color_palette("hls", cl),
      data=df.loc[rndperm,:],
      legend="full",
      alpha=0.3
  )

def Three_D_visualization(df,t):  
  print("3-D visualization:")
  np.random.seed(42)
  rndperm = np.random.permutation(df.shape[0])

  if t:
    A=df.loc[rndperm,:]["y"]
  else :
    A=None

  ax = plt.figure(figsize=(16,10)).gca(projection='3d')
  ax.scatter(
      xs=df.loc[rndperm,:]["pca-one"], 
      ys=df.loc[rndperm,:]["pca-two"], 
      zs=df.loc[rndperm,:]["pca-three"], 
      c=A, 
      cmap='tab10'
  )
  ax.set_xlabel('pca-one')
  ax.set_ylabel('pca-two')
  ax.set_zlabel('pca-three')
  plt.show()
two_D_visualization(df,0,0)
Three_D_visualization(df,0)

def plotBar(l):
  
  print(l)
  l.sort()
  # print(l)
  x=[x for x in range(len(l))]
  dct={}
  dct['Cluster number']=x
  dct['Size of cluster']=l
  # print(x,l)
  dtf=pd.DataFrame(dct)
  sns.barplot(x='Cluster number', y='Size of cluster', data=dtf)
  plt.show()

from collections import Counter
def Kmeans(df,c,true_labels,clf):
  # c=7

  kmeans = KMeans(n_clusters=c, random_state=0,init=clf,max_iter=100).fit(df)
  # print(kmeans.labels_)
  df['y']=kmeans.labels_
  count = Counter(df['y'])
  l=list(count.values())
  if(clf=='random'):
    print("Bar graph for the no of clusters:",c, "by using K-Means clustering Algorithm.")
  elif(clf=='k-means++'):

    print("Bar graph for the no of clusters:",c ,"by using K-Means++ clustering Algorithm.")
    if(c==7):
      df1=pd.concat([data['id'],df['y']],axis=1)
      df1.to_csv("result.cvs",index=False)

      
  plotBar(l)
  two_D_visualization(df,1,c)
  Three_D_visualization(df,1)
  

  print("Centroid of clusters: ")
  print(kmeans.cluster_centers_)


true_labels=[540, 542,  743, 540, 540, 675,  540,]
print("Bar graph for the given true values:")
plotBar(true_labels)

C=[6,7,10]
clfs=['random','k-means++']


for clf in clfs:
    for c in C:
      Kmeans(df.copy(),c,true_labels,clf)

from sklearn.cluster import Birch
def Birch_clustering(df,c): 
  clf = Birch(n_clusters=c).fit(df)
  labels = clf.predict(df)

  df['y']=labels
  count = Counter(labels)
  l=list(count.values())
  # if(clf=='random'):
  print("Bar graph for the no of clusters:",c, "by using Birch clustering Algorithm.")
  # elif(clf=='k-means++'):
  #   print("Bar graph for the no of clusters:",c ,"by using K-Means++ clustering Algorithm.")
  plotBar(l)
  two_D_visualization(df,1,c)
  Three_D_visualization(df,1)

C=[6,7,10]
for c in C:
  Birch_clustering(df.copy(),c)

def runnerfunction():
  # 
  # test_csv_path=input()
  # change this path to test.csv path
  data1=pd.read_csv("clustering_data.csv")
  df=preprocessing(data1)
  Kmeans(df.copy(),7,true_labels,'k-means++')


  

# ##############################
# runner function
runnerfunction()
