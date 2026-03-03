import shap
import numpy as np
import matplotlib.pyplot as plt

def f(model, X):
    return model.predict(X).flatten()



def generate_explanation(model, X_train, X_test, explainer_type='tree'):
    """
    Generate SHAP explanation for the model predictions.
    
    Parameters:
    - model: The trained model (e.g., XGBoost, LightGBM).
    - X_train: Training feature set.
    - X_test: Test feature set.
    - explainer_type: Type of SHAP explainer to use ('tree', 'linear', etc.).
    
    Returns:
    - explainer: SHAP explainer object.
    - explanation: SHAP explanation object.
    - shap_values: SHAP values for the test set.
    """
    
    if explainer_type == 'tree':
        explainer = shap.TreeExplainer(model.predict, X_train)
    elif explainer_type == 'NN':
        explainer = shap.KernelExplainer(model.predict, X_train)
    else:
        raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    shap_values = explainer.shap_values(X_test)
    explanation = explainer(X_test)
    
    return explainer, explanation, shap_values

def plot_shap_summary(explanation, X_test, first_n=10, plot_type=None):

    max_abs_shap = np.abs(explanation.values).mean(axis=0)
    top_inds = np.argsort(max_abs_shap)[-first_n:]
    X_display = X_test.iloc[:, top_inds]

    shap.summary_plot(explanation, X_display)
