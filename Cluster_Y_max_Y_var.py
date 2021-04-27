import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random as rd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from math import sqrt


X, y = make_blobs(n_samples=300, centers=2)
dataset = pd.DataFrame(X, y)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
dataset = pd.DataFrame(X, index=list(range(len(X))))
dataset.columns = ['x', 'y']
dataset['cluster'] = kmeans.labels_

coords_of_centers = []
for i in kmeans.cluster_centers_:
    coords_of_centers.append(i)
centers = pd.DataFrame(coords_of_centers)
centers.columns = ['x', 'y']


def make_curve(data):
    """ Функция рассчитывает координаты точек (x,y) кривой через минимальное среднее евликодово расстояние 
    между граничными точками двух кластеров. Возвращает датафреймы: координаты точек 1го кластера, координаты точек
    2го кластера, координаты точек полученной кривой """
    def dist(x, y):
        """Функция рассчета евклидова расстояния"""  
        d=0
        for i in range(len(x)):
            d+=(x[i]-y[i])**2
        return sqrt(d)  
    
    cluster_1, cluster_2, coord_x, coord_y = [], [], [], []
    
    for ind, row in data.iterrows():
        if row[2] == 0:
            cluster_1.append((row[0], row[1]))
        else:
            cluster_2.append((row[0], row[1]))
            
    for i,j in cluster_1:
        m = 0.9
        for n, l in cluster_2:
            d = dist((i,j), (n,l))
            if d <= m:
                x = (i+n)/2
                y = (j+l)/2
                coord_x.append(x)
                coord_y.append(y)
                
    df_1 = pd.DataFrame(cluster_1)
    df_2 = pd.DataFrame(cluster_2)
    df_1.columns = ['x', 'y']
    df_2.columns = ['x', 'y']
    df_x = pd.DataFrame(coord_x)
    df_y = pd.DataFrame(coord_y)
    df_coord = pd.concat([df_x, df_y], axis=1)
    df_coord.columns = ['x', 'y']
    return df_1, df_2, df_coord

df_1, df_2, df_coord = make_curve(dataset)


def show_plot(data1, data2, data3, data, name_of_plot):
    if name_of_plot == 'scatter':
        plt.scatter(data1['x'], data1['y'], c='red')
        plt.scatter(data2['x'], data2['y'], c='yellow')
        plt.scatter(data3['x'], data3['y'], c='blue')
    if name_of_plot == 'scatter with reg':
        fig, ax = plt.subplots(figsize= (15, 10))
        sns.scatterplot(data=data1, x="x", y="y",color='red', s=50, ax=ax)
        sns.scatterplot(data=data2, x="x", y="y", color='yellow', s=50, ax=ax)
        sns.regplot(data=data3, x="x", y="y", order=4.9, truncate=True, ci=None, scatter=False, ax=ax)  
        # order = 1 (прямая) => order - ...
    sns.scatterplot(data=data, x="x", y="y", color='green', marker = 'X', s = 70, ax=ax)
    plt.show()

show_plot(df_1, df_2, df_coord, centers, 'scatter with reg')


# ## Y_max, Y_Var, Random

# для Y_var: RFclassifier => predict_proba - матрица из 2х столбцов: 0 и 1, 
#     в каждом столбце показывается уверенность к какому классу она относится.
#     Отсчеь только первый столбец (1), не нулевой. (слайсы)
# Чем ближе 1, тем выше уверенность. Нам нужно 0,5. p * (1-p) - максимум в 0,5. p - значение первого столбца. 
# Если я нахожу максимум этого выражения, то находятся объекты, которые нам нужны. 


n_cluster = 2
d = [[] for i in range(n_cluster)]  # создается список пустых списков, количество списоков = n_cluster
for j in range(n_cluster):
    for n, el in enumerate(kmeans.labels_):
        if el == j:
            d[j].append(n)
random_numbers = [rd.choice(x) for x in d]
print(random_numbers)

part_training_set, other_training_set = [], []
for n, x in enumerate(X):
    if n in random_numbers:
        part_training_set.append(x)
    else:
        other_training_set.append(x)


# RFClassifier
X_train = [x for x in part_training_set]
Y_train = [j for i, j in enumerate(kmeans.labels_) if i in random_numbers]
X_test = [x for x in other_training_set]
Y_test = [j for i, j in enumerate(kmeans.labels_) if i not in random_numbers]
model = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=1, n_jobs=-1).fit(
                X_train, Y_train)
model.score(X_test, Y_test)  # 0.9093959731543624


# Y_max
Y_pred_test = model.predict(X_test)
value = pd.Series(Y_pred_test).sort_values(ascending=False).head(20).index.sort_values(ascending=False)
part_training_set_copy = part_training_set.copy()
for x in value:
    X_max = other_training_set[x]
    part_training_set_copy.append(X_max)
df_Y_max = pd.DataFrame(part_training_set_copy)
df_Y_max.columns = ['x', 'y']
df_Y_max['method'] = 'Y_MAX'


# Y_var
Y_pred = model.predict_proba(X_test)
df_y_pred = pd.DataFrame(Y_pred)
cluster_1 = list(df_y_pred[1])
val = []
while len(val) != 20:
    val_max = [0, 0, 0]
    for i, p in enumerate(cluster_1):
        f = p * (1 - p)
        if f > val_max[2]:
            val_max = [i, p, f]
    val.append(val_max[0])
    cluster_1[val_max[0]] = 0

X_var = []
for x in val:
    var = other_training_set[x]
    X_var.append(var)
df_Y_var = pd.DataFrame(X_var)
df_Y_var.columns = ['x', 'y']
df_Y_var['method'] = 'Y_VAR'

# Concat dataframes
df = pd.concat([df_Y_max, df_Y_var])
# Show plot
fig, ax = plt.subplots(figsize= (15, 10))
markers = {'Y_VAR': 's', "Y_MAX": 'X'}
sns.scatterplot(data=df_1, x="x", y="y",color='green', s = 50, ax=ax)
sns.scatterplot(data=df_2, x="x", y="y", color='yellow', s = 50, ax=ax)
sns.regplot(data=df_coord, x="x", y="y", order=4.9, truncate=True, ci=None, scatter=False, ax=ax)
sns.scatterplot(data=centers, x="x", y="y", color='red', s = 90, marker = 'v', ax=ax)
sns.scatterplot(data=df, x="x", y="y", palette = ['blue', 'orange'], hue = 'method', markers = markers,style='method', s = 50, 
                alpha =0.8, ax=ax)
plt.setp(ax.get_legend().get_texts(), fontsize='10') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
plt.grid(True)
plt.show()





