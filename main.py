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
import xgboost as xgb
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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

def aggregate_features(patient_samples):
    stats = []
    for feature_idx in range(15):  # 遍历每个特征
        feature_values = patient_samples[:, feature_idx]
        stats.extend([
            np.mean(feature_values),
            np.std(feature_values),
            np.median(feature_values),
            np.min(feature_values),
            np.max(feature_values),
        ])
    return stats

if __name__ == '__main__':
    raw_X_list, raw_Y_list = [], []
    useUmap = True
    chunk_length = 10000
    if not useUmap:
        for root, dirs, files in os.walk('Data/DataInPatients'):
            for file in files:
                if 'npy' in file:
                    print('Proceeding {}...'.format(file))
                    numpy_data = np.load(os.path.join(root, file)) / 1023.  # standarize
                    raw_X_list.append(numpy_data)
                    raw_Y_list.append(int(file.split('_')[-1].split('.')[0]))

        # Figure1(raw_X_list, raw_Y_list)
        # PCA(raw_X_list, raw_Y_list)
        # Figure3(raw_X_list, raw_Y_list)

        X_aggregated = np.array([aggregate_features(patient) for patient in raw_X_list])
        y = np.array(raw_Y_list)
    
    else:
        for root, dirs, files in os.walk('Data/DataInPatientsUmap'):
            for file in files:
                if 'npy' in file:
                    print('Proceeding {}...'.format(file))
                    numpy_data = np.load(os.path.join(root, file)) / 1023.  # standarize
                    raw_X_list.append(numpy_data)
                    raw_Y_list.append(int(file.split('_')[-1].split('.')[0]))

    
    # 比较纵向特征聚合和横向特征聚合+XGBoost的效果
    

    # Define your filtered data and labels
    S = X_aggregated  # Replace with your actual filtered data
    t = y  # Replace with your actual labels

    # Set up the XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,       # 1,000 trees
        max_depth=6,             # Maximum depth of trees
        learning_rate=0.01,      # Learning rate
        subsample=0.8,           # Subsample to prevent overfitting
        colsample_bytree=1,      # Column sampling
        use_label_encoder=False, # Avoid deprecation warning
        eval_metric='logloss'    # Evaluation metric
    )

    # Define the hyperparameter grid for the inner loop (GridSearchCV)
    param_grid = {
        'n_estimators': [100, 300, 500, 700, 1000, 2000],
        'max_depth': [3, 6, 8, 10, 12],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1]
    }

    # Set up StratifiedKFold for the outer loop (for nested cross-validation)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Set up StratifiedKFold for the inner loop (for hyperparameter tuning)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6))

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(S, t)):
        # Split the data into training and testing sets
        X_train, X_test = S[train_idx], S[test_idx]
        y_train, y_test = t[train_idx], t[test_idx]
        
        # Perform GridSearchCV for hyperparameter tuning on the inner loop
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=inner_cv, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_
        
        # Get predicted probabilities for ROC AUC computation
        y_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Compute the ROC curve
        viz = RocCurveDisplay.from_estimator(
            best_model,
            X_test,
            y_test,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == outer_cv.get_n_splits() - 1),  # Fixing issue here
        )
        
        # Compute AUC for the current fold
        fold_auc = roc_auc_score(y_test, y_proba)
        aucs.append(fold_auc)
        
        # Interpolate the true positive rate
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Calculate the mean AUC across all folds
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Calculate the mean TPR and plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # Plot the standard deviation band
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    # Set plot labels and legend
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve with Nested Cross-Validation"
    )

    ax.legend(loc="lower right")

    # Save and show the plot
    plt.savefig('Results/XGB/ROC_nested_cv.png', dpi=900)
    plt.show()

    # Print the average AUC across folds
    print(f"Average AUC across folds: {mean_auc:.2f} ± {std_auc:.2f}")