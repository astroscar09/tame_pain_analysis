from sklearn.utils.class_weight import compute_class_weight

#This can be used to compute class weights for handling class imbalance. I need to double check the pain responses and see if they are balanced.
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_tensor.numpy()), y=y_train_tensor.numpy())
weights = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)