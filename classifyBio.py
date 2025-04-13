from readBioChe import read_excel_with_pandas, getSomeCols
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import pandas as pd
import seaborn as sns
from itertools import product


if __name__ == '__main__':
    raw_X_list, raw_Y_list = [], []
    file_path = "Data/M2M5BIOCHE.xlsx"  # 替换为实际文件路径
    data = read_excel_with_pandas(file_path, sheet_name='Sheet2')
    X = getSomeCols(data, cols=['ALT', 'AST', '总胆红素', '白蛋白', '球蛋白', '肌酐', 'Ccr', 'K', 'Ca', 'P', 'Na', '尿酸', '甘油三酯', '胆固醇', 'LDL', 'HDL', '尿素']).to_numpy()
    Y = getSomeCols(data, cols=['分型']).to_numpy().flatten().astype(int)
    
    # Define your filtered data and labels
    S = X  # Replace with your actual filtered data
    t = Y  # Replace with your actual labels

    # 添加类别权重计算
    class_weights = len(t) / (2 * np.bincount(t))
    weight_ratio = class_weights[1] / class_weights[0]  # 适用于二分类问题

    # 优化后的参数空间
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.05, 0.1, 0.2],
    }

    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['learning_rate']
    ))
    param_combinations = pd.DataFrame(
        param_combinations,
        columns=['n_estimators', 'max_depth', 'learning_rate']
    )

    auc_means, auc_stds = [], []
    for params in param_combinations.itertuples(index=False):
        estimator_num = params.n_estimators
        depth = params.max_depth
        lr = params.learning_rate

        # 优化后的模型配置
        xgb_model = xgb.XGBClassifier(
            n_estimators=estimator_num,           # 减少基础树数量
            max_depth=depth,                # 降低默认深度
            learning_rate=lr,          # 提高基础学习率
            subsample=0.7,              # 降低子采样率
            colsample_bytree=0.8,       # 增加特征采样正则化
            reg_alpha=0,              # 添加L1正则化
            reg_lambda=1,               # 添加L2正则化
            scale_pos_weight=weight_ratio,  # 处理类别不平衡
            eval_metric='logloss',
            random_state=42             # 固定随机种子
        )

        # Set up StratifiedKFold for the outer loop (for nested cross-validation)
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(S, t), total=outer_cv.get_n_splits(), desc='CV Progress')):
            # Split the data into training and testing sets
            X_train, X_test = S[train_idx], S[test_idx]
            y_train, y_test = t[train_idx], t[test_idx]
            
            # Perform GridSearchCV for hyperparameter tuning on the inner loop
            xgb_model.fit(X_train, y_train)
            
            # Get predicted probabilities for ROC AUC computation
            y_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Compute AUC for the current fold
            fold_auc = roc_auc_score(y_test, y_proba)
            aucs.append(fold_auc)

        # Calculate the mean AUC across all folds
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        auc_means.append(mean_auc)
        auc_stds.append(std_auc)

    # 创建绘图DataFrame
    plot_df = pd.DataFrame({
        "AUC_mean": auc_means,
        "AUC_std": auc_stds,
    })
    plot_df = pd.concat([plot_df, param_combinations], axis=1)
    param_ids = [f"P{i+1}" for i in range(len(plot_df))]
    plot_df["param_id"] = param_ids

    # 设置绘图样式
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(16, 8))  # 原为 (12,8)，增加宽度

    # 调整主图区域右侧边距
    plt.subplots_adjust(right=0.65)  # 原为 0.75，减少右侧空间

    # 绘制点图 + 误差条
    ax = sns.scatterplot(
        data=plot_df,
        x="AUC_mean", 
        y="param_id",
        hue="learning_rate",  # 用颜色区分学习率
        size="max_depth",     # 用点大小区分树深度
        sizes=(50, 200),     # 大小范围
        palette="viridis",    # 颜色映射
        edgecolor="black",
        linewidth=0.5,
    )

    # 添加误差条
    plt.errorbar(
        x=plot_df["AUC_mean"],
        y=plot_df["param_id"],
        xerr=plot_df["AUC_std"],
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
    )

    # 添加参数表格注释
    param_table = plt.table(
    cellText=plot_df[["n_estimators", "max_depth", "learning_rate"]].values,
    colLabels=["Estimators", "Depth", "LR"],  # 缩短列名
    rowLabels=plot_df["param_id"],
    loc="right",
    bbox=[1.05, -0.0, 0.25, 0.8],  # 调整位置 [x, y, width, height]
    cellLoc="center",
    colWidths=[0.15, 0.15, 0.15]  # 控制列宽
    )

    # 设置表格字体
    param_table.auto_set_font_size(False)
    param_table.set_fontsize(10)  # 调小字体

    # 调整行高
    for key, cell in param_table.get_celld().items():
        cell.set_height(0.05)  # 行高调整

    # 美化标签和标题
    ax.set_xlabel("AUC (Mean ± Std)", fontweight="bold")
    ax.set_ylabel("Parameter Combination ID", fontweight="bold")
    ax.set_yticklabels(
        plot_df["param_id"],
        rotation=0,  # 0度水平显示
        ha="right",  # 对齐方式
        fontsize=10,   # 调小标签字体
        fontweight="bold"
    )
    plt.title("XGBoost Parameter Tuning Results (K-Fold Cross-Validation)", pad=20, fontsize=14)
    # 将图例移动到图表左上方
    plt.legend(
        bbox_to_anchor=(1, 1.05),  # 调整位置
        loc="upper left",
        title="Hyperparams",
        frameon=False,  # 去除边框
        ncol=2  # 分两列显示
    )

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.77)  # 为表格留出空间
    plt.savefig('Results/XGB/BIOCHE/GridAUC.png', dpi=800)