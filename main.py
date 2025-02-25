import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.array as da
from dask.distributed import Client
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pycirclize import Circos

os.environ["OMP_NUM_THREADS"] = "2"
features_name = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR']

def parallel_silhouette_score(data, labels, batch_size=1000):
    n_samples = len(data)
    batches = [slice(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]
    # 定义延迟任务
    tasks = [dask.delayed(silhouette_samples)(data[slice_obj], labels[slice_obj]) for slice_obj in batches]
    # 执行所有任务
    results = dask.compute(*tasks)
    # 将结果拼接成一个数组
    all_scores = np.concatenate(results)
    # 计算平均轮廓系数
    silhouette_avg = np.mean(all_scores)
    
    return silhouette_avg

def parallel_calinski_harabasz_score(dask_data, labels):
    dask_data = da.from_array(dask_data, chunks=(100000, -1))
    result = calinski_harabasz_score(dask_data.compute(), labels)
    return result

def parallel_davies_bouldin_score(dask_data, labels):
    dask_data = da.from_array(dask_data, chunks=(100000, -1))
    result = davies_bouldin_score(dask_data.compute(), labels)
    return result

def plotProtein(name_1, name_2, X, Y):
    index_1 = features_name.index(name_1)
    index_2 = features_name.index(name_2)
    proteins_array = []
    labels_array = []
    # 取出需要的两列蛋白
    for i in range(len(X)):
        proteins_data = X[i][:, [index_1, index_2]]
        proteins_array.append(proteins_data)
        if Y[i] == 0:
            labels = np.repeat(0, len(X[i]))
        else:
            labels = np.repeat(1, len(X[i]))
        labels = labels.reshape(-1, 1)
        labels_array.append(labels)

    proteins_array = np.vstack(proteins_array)
    labels_array = np.vstack(labels_array)
    
    proteins_df = pd.DataFrame(proteins_array, columns=[name_1, name_2])
    proteins_df['Label'] = labels_array
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=name_1, y=name_2, hue='Label', data=proteins_df, palette={0: 'blue', 1: 'green'}, s=1)
    # 更新刻度标签字体大小
    plt.xlabel(name_1, fontweight='bold')  # 横坐标轴名字加粗
    plt.ylabel(name_2, fontweight='bold')  # 纵坐标轴名字加粗
    plt.legend(markerscale=5)
    plt.savefig('Results/Proteins/{}VS{}'.format(name_1, name_2), dpi=600)
    print('Results/Proteins/{}VS{}.png is SAVED'.format(name_1, name_2))

def Figure1(raw_X_list, raw_Y_list):
    protein_pair = [['CD45', 'CD34'], ['CD45', 'SSC-A'], ['CD13', 'CD33'], ['HLA-DR', 'CD117'], ['CD56', 'CD7']]
    for i in range(len(protein_pair)):
        plotProtein(protein_pair[i][0], protein_pair[i][1], raw_X_list, raw_Y_list)

def PCA(X_list, Y_list, n_components=2):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from scipy.stats import chi2
    from sklearn.preprocessing import StandardScaler

    labels_array = []
    for i in range(len(X_list)):
        if Y_list[i] == 0:
            labels = np.repeat(0, len(X_list[i]))
        else:
            labels = np.repeat(1, len(X_list[i]))
        labels = labels.reshape(-1, 1)
        labels_array.append(labels)

    data = np.vstack(X_list)
    labels_array = np.vstack(labels_array)
    # 归一化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # 使用PCA进行降噪
    pca = PCA(n_components=n_components)
    pca.fit(data)
    reduced_data = pca.transform(data)

    df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
    df['Label'] = labels_array
    # 绘制降噪后的数据
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Label', data=df, palette={0: 'blue', 1: 'green'}, s=1)
    # 更新刻度标签字体大小
    plt.xlabel('PCA1', fontweight='bold')  # 横坐标轴名字加粗
    plt.ylabel('PCA2', fontweight='bold')  # 纵坐标轴名字加粗
    plt.legend(markerscale=5)
    plt.savefig('Results/PCA_Results/PCA'.format('PCA1', 'PCA2'), dpi=600)
    print('Results/PCA_Results/PCA.png is SAVED'.format('PCA1', 'PCA2'))

def KMeans(X_train, K, draw=False):
    import matplotlib.pyplot as plt
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from scipy.spatial.distance import cdist
    from sklearn.datasets import load_iris
    from sklearn.metrics import silhouette_samples
    import joblib

    distortions = []
    sses = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    X_train = X_train[: 11480000]
    
    for k in K:
        print('=================================== K = {} =========================================='.format(k))
        #分别构建各种K值下的聚类器
        Model = MiniBatchKMeans(n_clusters=k, n_init='auto', verbose=1, batch_size=100000)
        Model.fit(X_train)
        if len(K) == 1:
            # 保存模型
            joblib.dump(Model, 'Results/Kmeans_Curve/kmeans_model.skops')
            return Model

        sses.append(Model.inertia_)
        silhouette_scores.append(parallel_silhouette_score(X_train, Model.labels_))
        calinski_harabasz_scores.append(parallel_calinski_harabasz_score(X_train, Model.labels_))
        davies_bouldin_scores.append(parallel_davies_bouldin_score(X_train, Model.labels_))

    plt.plot(K, sses, label='WCSS', marker='o', linestyle='-', linewidth=1)
    plt.xlabel('optimal K')
    plt.ylabel('WCSS')
    plt.savefig('Results/Kmeans_Curve/2-20.png')
    plt.close()

    plt.plot(K, silhouette_scores, label='silhouette', marker='o', linestyle='-', linewidth=1)
    plt.xlabel('optimal K')
    plt.ylabel('silhouette_score')
    plt.savefig('Results/Kmeans_Curve/2-20_silhouette_score.png')
    plt.close()

    plt.plot(K, calinski_harabasz_scores, label='calinski_harabasz', marker='o', linestyle='-', linewidth=1)
    plt.xlabel('optimal K')
    plt.ylabel('calinski_harabasz_score')
    plt.savefig('Results/Kmeans_Curve/2-20_calinski_harabasz_score.png')
    plt.close()

    plt.plot(K, davies_bouldin_scores, label='davies_bouldin', marker='o', linestyle='-', linewidth=1)
    plt.xlabel('optimal K')
    plt.ylabel('davies_bouldin_score')
    plt.savefig('Results/Kmeans_Curve/2-20_davies_bouldin_score.png')
    plt.close()

def plotHistogram(X, model, n_clusters, class_name):
    # Analyze clustering: Count number of cells per cluster for each patient
    n_patients = len(X)
    cluster_counts = np.zeros((n_patients, n_clusters), dtype=int)

    for i in range(len(X)):
        # Count number of cells in each cluster for this patient
        patient_cluster_labels = model.predict(X[i])
        unique, counts = np.unique(patient_cluster_labels, return_counts=True)
        cluster_counts[i, unique] = counts

    # Convert cluster counts to a DataFrame for easier visualization
    cluster_counts_df = pd.DataFrame(cluster_counts.T, index=[f"Cluster {i+1}" for i in range(n_clusters)],
                                    columns=[f"Patient {i+1}" for i in range(n_patients)])

    circos = Circos.initialize_from_matrix(
        cluster_counts_df,
        space=2,
        r_lim=(93, 100),
        cmap='rainbow',
        label_kws=dict(r=105, size=9, orientation='vertical', weight='bold', color="black"))
    fig = circos.plotfig()
    plt.savefig('Results/Circos/{}ClustersCircos.png'.format(class_name), dpi=600)
    plt.close()

def Figure3(raw_X_list, raw_Y_list):
    # for Circos and Kmeans
    X = np.vstack(raw_X_list)
    K = 17
    model = KMeans(X, [K])  # K = 17
    # 分别对两类病人做circos分析
    M2_X = []
    M5_X = []
    for i in range(len(raw_X_list)):
        if raw_Y_list[i] == 0:
            M2_X.append(raw_X_list[i])
        else:
            M5_X.append(raw_X_list[i])
    plotHistogram(M2_X, model, K, 'M2')
    plotHistogram(M5_X, model, K, 'M5')



if __name__ == '__main__':
    raw_X_list, raw_Y_list = [], []
    for root, dirs, files in os.walk('Data/DataInPatients'):
        for file in files:
            if 'npy' in file:
                print('Proceeding {}...'.format(file))
                numpy_data = np.load(os.path.join(root, file)) / 1023.  # standarize
                raw_X_list.append(numpy_data)
                raw_Y_list.append(int(file.split('_')[-1].split('.')[0]))

    # Figure1(raw_X_list, raw_Y_list)
    # PCA(raw_X_list, raw_Y_list)
    Figure3(raw_X_list, raw_Y_list)