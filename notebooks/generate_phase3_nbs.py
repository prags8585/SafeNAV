import json

base_imports = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc\n",
    "from sklearn.preprocessing import label_binarize\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'figure.autolayout': True})\n",
    "\n",
    "print(\"Loading preprocessed (imbalanced) dataset from Phase 2...\")\n",
    "df = pd.read_csv('../outputs/processed_imbalanced_data.csv')\n",
    "print(f\"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.\")"
]

train_test_split_code = [
    "X = df.drop('Severity', axis=1)\n",
    "y = df['Severity']\n",
    "\n",
    "# 80% for training, 20% for testing\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(f\"Training set: {X_train.shape[0]:,} rows.\")\n",
    "print(f\"Testing set:  {X_test.shape[0]:,} rows.\")"
]

def make_notebook(is_smote):
    cells = []
    
    # Title
    title = "# Phase 3b: Decision Tree Model Implementation (Balanced with SMOTE)" if is_smote else "# Phase 3a: Decision Tree Model Implementation (Imbalanced Data)"
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [title]})
    
    # Imports
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": base_imports})
    
    # Split
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Step 1: Train/Test Split"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": train_test_split_code})
    
    if is_smote:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Step 2: Apply SMOTE to Training Data ONLY"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "print(\"Applying SMOTE to training data ONLY...\")\n",
            "smote = SMOTE(random_state=42)\n",
            "X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)\n",
            "\n",
            "print(f\"Training set (After SMOTE): {X_train_balanced.shape[0]:,} rows.\")"
        ]})
    
    # Train
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Train the Model"]})
    if is_smote:
        train_code = [
            "print(\"Training Decision Tree Model on balanced data...\")\n",
            "dt_model = DecisionTreeClassifier(max_depth=15, random_state=42)\n",
            "dt_model.fit(X_train_balanced, y_train_balanced)\n",
            "print(\"Training Complete!\")"
        ]
    else:
        train_code = [
            "print(\"Training Decision Tree Model on raw imbalanced data...\")\n",
            "dt_model = DecisionTreeClassifier(max_depth=15, random_state=42)\n",
            "dt_model.fit(X_train, y_train)\n",
            "print(\"Training Complete!\")"
        ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": train_code})
    
    # Evaluate
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Evaluate on the UNTOUCHED Test Set"]})
    eval_code = [
        "y_pred = dt_model.predict(X_test)\n",
        "y_prob = dt_model.predict_proba(X_test)\n",
        "\n",
        "print(\"--- Decision Tree Evaluation Metrics ---\")\n",
        "print(f\"Accuracy:  {accuracy_score(y_test, y_pred):.4f}\")\n",
        "print(f\"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}\")\n",
        "print(f\"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}\")\n",
        "print(f\"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}\")"
    ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": eval_code})
    
    # CM
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Confusion Matrix"]})
    cm_code = [
        "plt.figure(figsize=(8, 6))\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])\n",
        "plt.title('Decision Tree Confusion Matrix (Test Set)', fontsize=16, pad=20)\n",
        "plt.ylabel('Actual Severity', fontsize=12)\n",
        "plt.xlabel('Predicted Severity', fontsize=12)\n",
        "plt.show()"
    ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cm_code})
    
    # ROC
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### ROC Curve (Predicting Fatal Accidents)"]})
    roc_code = [
        "y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4])\n",
        "fatal_class_idx = 3 # Index for Severity 4\n",
        "\n",
        "fpr, tpr, _ = roc_curve(y_test_bin[:, fatal_class_idx], y_prob[:, fatal_class_idx])\n",
        "roc_auc = auc(fpr, tpr)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate (Recall)')\n",
        "plt.title('ROC Curve (Predicting Fatal Accidents)')\n",
        "plt.legend(loc=\"lower right\")\n",
        "plt.show()"
    ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": roc_code})
    
    # Feature Imp
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Feature Importances"]})
    feat_code = [
        "importances = dt_model.feature_importances_\n",
        "indices = np.argsort(importances)[::-1][:15] # Top 15\n",
        "features = X.columns\n",
        "\n",
        "plt.figure(figsize=(10, 8))\n",
        "plt.title(\"Top 15 Most Important Features (Decision Tree)\")\n",
        "plt.barh(range(len(indices)), importances[indices], color=\"b\", align=\"center\")\n",
        "plt.yticks(range(len(indices)), [features[i] for i in indices])\n",
        "plt.xlabel(\"Relative Importance\")\n",
        "plt.gca().invert_yaxis()\n",
        "plt.show()"
    ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": feat_code})
    
    # Tree
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Decision Tree Structure (Top 3 Levels)"]})
    tree_code = [
        "plt.figure(figsize=(20, 10))\n",
        "plot_tree(dt_model, max_depth=3, feature_names=X.columns, class_names=['1', '2', '3', '4'], filled=True, rounded=True, fontsize=10)\n",
        "plt.title(\"Decision Tree Structure (Top 3 Levels)\", fontsize=20)\n",
        "plt.show()"
    ]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": tree_code})
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

with open('03a_decision_tree_imbalanced.ipynb', 'w') as f:
    json.dump(make_notebook(False), f, indent=1)
    
with open('03b_decision_tree_balanced_smote.ipynb', 'w') as f:
    json.dump(make_notebook(True), f, indent=1)

