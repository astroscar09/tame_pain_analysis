from xgboost import XGBClassifier
import xgboost as xgb 
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score


params = {'eta': 0.3, #default learning rate and can change to be between [0-1]
          'nestimators': 100,
          'max_depth': 6, #default max depth of the tree this regulates the complexity of the model 
          'learning_rate': 0.1,
          'subsample': 0.7, #default subsample ratio of the training instance, a value of 0.5 means that half of the training data will be used to grow trees useful for preventing overfitting (0, 1]
          'lambda': 1, #default L2 regularization term on weights, can be used to avoid overfitting (0, inf)
          'alpha': 0, #default L1 regularization term on weights, can be used to avoid overfitting (0, inf)
          'sample_method': 'subsample', #default sampling method for the training data. Can be 'uniform': every data is equally likely to make it, 'subsample': randomly subsample X% of the data, 'gradient_based'
          'tree_method': 'auto', #default tree construction algorithm, can be 'auto', 'exact', 'approx', 'hist',
           'eval_metric': 'error', 
            'seed': 42, #default random seed for reproducibility}
            
        }

def load_features():
    
    df = pd.read_csv('Merged_Features.csv')

    X = df.drop(columns=['Pain'])
    y = df['Pain']

    return X, y


X, y = load_features()


#data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.2, random_state=42)


def explore_hyperparameters(X_train, y_train):

    param_dist = {
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'n_estimators': randint(50, 300),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                 }


def train_model(X_train, y_train):

    bst = XGBClassifier(
        objective='binary:logistic',   # Required for binary classification
        eval_metric='error',         # Or 'auc' / 'error'
        n_estimators=100,              # not 'nestimators'
        max_depth=6,
        learning_rate=0.1,             # alias for 'eta'
        subsample=0.7,
        reg_lambda=1,                  # not 'lambda'
        reg_alpha=0,                   # not 'alpha'
        tree_method='auto',
        random_state=42
    )

    # fit model
    bst.fit(X_train, y_train)

    return bst

def test_model(bst, X_test):

    """
    Tests the trained model on the test dataset and makes predictions.
    Parameters:
    bst (XGBClassifier): The trained XGBoost model.
    X_test (DataFrame): The test dataset features.
    Returns:
    preds (array): The predicted labels for the test dataset.
    """
    # ensure the model is trained

    # make predictions
    y_pred = bst.predict(X_test, output_margin=True)

    return y_pred

def compute_shap_values(bst, X_train):


    explainer = shap.TreeExplainer(bst)
    explanation = explainer(X_train)

    shap_values = explanation.values

    pred = test_model(bst, X_train)

    print(np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max())

    return explanation, shap_values

def plot_shap_summary(X_train, shap_values, first_n = 10, plot_type = None):

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_inds = np.argsort(mean_abs_shap)[-first_n:]

    X_display = X_train.iloc[:, top_inds]
    shap_display = shap_values[:, top_inds]

    shap.summary_plot(shap_display, X_display, plot_type=plot_type)


def plot_importance_by_feature(shap_values, feature_idx):

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[feature_idx])

def plot_force_plot(shap_values, feature_idx):

    # visualize the first prediction's explanation with a force plot
    shap.plots.force(shap_values[feature_idx])

def plot_shap_bar(shap_values, first_n = None):

    if first_n is not None:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_inds = np.argsort(mean_abs_shap)[-first_n:]
        shap_display = shap_values[:, top_inds]
        shap.plots.bar(shap_display)

    else:
        shap.plots.bar(shap_values)



def plot_importance(bst):

    xgb.plot_importance(bst)

def plot_tree(bst, num_trees = 2, ipython = False):

    if ipython:
        xgb.to_graphviz(bst, num_trees=2)
    else:
        xgb.plot_tree(bst, num_trees=num_trees)    


def plot_shap_beeswarm(explanation):

    shap.plots.beeswarm(explanation)


def computing_metrics(y_test, y_pred):

    """
    Computes and prints the accuracy, AUC, and F1 score of the model predictions.
    Parameters:
    y_test (array-like): True labels of the test dataset.
    y_pred (torch.Tensor): Model predictions.
    """
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1_scor = f1_score(y_test, y_pred, average='micro')

    # Print the results
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUC: {auc * 100:.2f}%")
    print(f"Test F1 Score: {f1_scor * 100:.2f}%")