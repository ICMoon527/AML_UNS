import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Define your filtered data and labels
S = X  # Replace with your actual filtered data
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
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1]
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
plt.savefig('ROC_nested_cv.png', dpi=900)
plt.show()

# Print the average AUC across folds
print(f"Average AUC across folds: {mean_auc:.2f} ± {std_auc:.2f}")