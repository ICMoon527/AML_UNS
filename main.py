import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features_name = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR']

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

if __name__ == '__main__':
    raw_X_list, raw_Y_list = [], []
    for root, dirs, files in os.walk('Data/DataInPatients'):
        for file in files:
            if 'npy' in file:
                print('Proceeding {}...'.format(file))
                numpy_data = np.load(os.path.join(root, file)) / 1023.  # standarize
                raw_X_list.append(numpy_data)
                raw_Y_list.append(int(file.split('_')[-1].split('.')[0]))

    Figure1(raw_X_list, raw_Y_list)