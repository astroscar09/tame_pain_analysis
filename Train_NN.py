from NN_Model import ComplexClassifier
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file):

    if '.txt' in file:
        data = pd.read_csv(file, sep=' ')
    elif '.csv' in file:
        data = pd.read_csv(file)

    #print(data.columns)

    X = data.drop(columns=['Pain']).values
    Y = data[['Pain']].values

    return X, Y

def get_unique_pids():

    data_df = pd.read_csv('Merged_Data.csv')
    unique_PID = data_df['PID_x'].unique()  # Ensure that the 'PID' column is unique
    
    return unique_PID

def get_filtered_index(meta_audio):

    filtered_indices = meta_audio[meta_audio['ACTION LABEL'].isin([0, 1, 2])].index.values
    return filtered_indices



def split_data_by_pid(X, y, pids, seed = 43):

    """
    Splits the dataset into training and test sets based on unique patient IDs.
    
    Parameters:
    X (array-like): Features of the dataset.
    y (array-like): Labels of the dataset.
    pid (array-like): Unique patient IDs.
    
    Returns:
    tuple: Training and test data features and labels.
    """

    data_df = pd.read_csv('Merged_Data.csv')

    #mask = data_df['PID_x'].isin(pid)  # Create a mask for the specified PIDs

    np.random.seed(seed)  # For reproducibility

    training_pid = np.random.choice(pids, size=int(len(pids) * 0.7), replace=False)  # Randomly select 70% of PIDs
    testing_pid = np.setdiff1d(pids, training_pid)  # The remaining 30% for testing

    training_mask = data_df['PID_x'].isin(training_pid)  # Create a mask for training PIDs
    testing_mask = data_df['PID_x'].isin(testing_pid)  # Create a mask for testing PIDs

    X_train = X[training_mask]
    y_train = y[training_mask]

    X_test = X[testing_mask]
    y_test = y[testing_mask]

    return X_train, y_train, X_test, y_test

def get_data_by_condition(X, y, data_df, condition = 'RW/LC'):
    """
    Filters the dataset based on a specific condition.
    
    Parameters:
    X (array-like): Features of the dataset.
    y (array-like): Labels of the dataset.
    condition (str): The condition to filter the dataset by.
    
    Returns:
    tuple: Filtered features and labels.
    """

    #data_df = pd.read_csv('Merged_Data.csv')
    mask_rw_lc = ((data_df['COND'] == 'RW') | (data_df['COND'] == 'LC')).values # Ensure that the condition is either RW or LC
    mask_rc_lw = ((data_df['COND'] == 'RC') | (data_df['COND'] == 'LW')).values # Ensure that the condition is either RC or LW
   
    if condition == 'RW/LC':

        X_train = X[mask_rw_lc]
        y_train = y[mask_rw_lc]

        X_test = X[mask_rc_lw]
        y_test = y[mask_rc_lw]

    elif condition == 'RC/LW':
        
        X_train = X[mask_rc_lw]
        y_train = y[mask_rc_lw]

      
        X_test = X[mask_rw_lc]
        y_test = y[mask_rw_lc]
       
    else:
        raise ValueError("Condition must be either 'RW/LC' or 'RC/LW'.")

    print(f'Number of samples in training set: {X_train.shape[0]}')
    print(f'Number of samples in test set: {X_test.shape[0]}')
    return X_train, y_train, X_test, y_test

    

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and test sets.
    
    Parameters:
    X (array-like): Features of the dataset.
    y (array-like): Labels of the dataset.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: Training and test data features and labels.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Standardizes the training and test data using StandardScaler.
    
    Parameters:
    X_train (array-like): Training data features.
    X_test (array-like): Test data features.
    
    Returns:
    tuple: Scaled training and test data.
    """
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def convert_to_tensors(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Converts the scaled training and test data to PyTorch tensors.
    
    Parameters:
    X_train_scaled (array-like): Scaled training data features.
    y_train (array-like): Training data labels.
    X_test_scaled (array-like): Scaled test data features.
    y_test (array-like): Test data labels.
    
    Returns:
    tuple: Tensors for training and test data features and labels.
    """
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze() #needed to add this since y_train is a 2D array
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).squeeze()
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_model(X_train_tensor, y_train_tensor):
    
    # Initialize the model, loss function, and optimizer
    input_size = X_train_tensor.shape[1]  # Get the number of input features
    model = ComplexClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)


    # Training loop
    num_epochs = 2500 # Adjust as needed
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def grab_model_predictions(model, X_tensor):
    """
    Generates predictions from the model for the given input tensor.
    
    Parameters:
    model (torch.nn.Module): The trained model.
    X_tensor (torch.Tensor): Input data features as a tensor.
    
    Returns:
    torch.Tensor: Model predictions.
    """
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted

def computing_metrics(y_test, y_pred):

    """
    Computes and prints the accuracy, AUC, and F1 score of the model predictions.
    Parameters:
    y_test (array-like): True labels of the test dataset.
    y_pred (torch.Tensor): Model predictions.
    """


    all_preds = y_pred.numpy()
    
    # Calculate accuracy
    acc = accuracy_score(y_test, all_preds)
    auc = roc_auc_score(y_test, all_preds)
    f1_scor = f1_score(y_test, all_preds, average='micro')

    # Print the results
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUC: {auc * 100:.2f}%")
    print(f"Test F1 Score: {f1_scor * 100:.2f}%")

def plot_confusion_matrix(y_true, y_pred, output_file='confusion_matrix.png'):
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def test_model(model, X_test_tensor, y_test_tensor):
    """
    Tests the trained model on the test dataset.
    
    Parameters:
    model (torch.nn.Module): The trained model.
    X_test_tensor (torch.Tensor): Test data features as a tensor.
    y_test_tensor (torch.Tensor): Test data labels as a tensor.
    
    Returns:
    float: The accuracy of the model on the test dataset.
    """
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test_tensor).sum().item()
        accuracy = correct / y_test_tensor.size(0)
    
    return accuracy

def change_input_data_run_model(X_train, X_test, y_train, y_test, confusion_matrix_plot):
    """
    A function to change the input data and run the model.
    
    Parameters:
    X (array-like): Features of the dataset.
    y (array-like): Labels of the dataset.
    
    Returns:
    None
    """
    # Split the data into training and test sets

    # Scale the data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Convert to tensors
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensors(X_train_scaled, y_train,
                                                                                       X_test_scaled, y_test)

    # Train the model
    model = train_model(X_train_tensor, y_train_tensor)

    # Test the model
    accuracy = test_model(model, X_test_tensor, y_test_tensor)
    print(f'Test Accuracy: {accuracy:.4f}')

    y_pred = grab_model_predictions(model, X_test_tensor)

    computing_metrics(y_test_tensor, y_pred)
    plot_confusion_matrix(y_test_tensor, y_pred, output_file=confusion_matrix_plot)

    return model


def main():
    # Read the data
    X, y = read_data('Merged_Features.csv')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Convert to tensors
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensors(X_train_scaled, y_train, X_test_scaled, y_test)

    # Train the model
    model = train_model(X_train_tensor, y_train_tensor)

    # Test the model
    accuracy = test_model(model, X_test_tensor, y_test_tensor)
    print(f'Test Accuracy: {accuracy:.4f}')


if __name__ == "__main__":

    main()