import torch
from setup import load_test_data
from utils import load_model, calculate_metrics
from sklearn.metrics import roc_auc_score

def evaluate_classifier(model_path, dataset_path):
    """
    Evaluate a classifier model on a dataset and return performance metrics.

    Parameters:
    - model_path: Path to the classifier model.
    - dataset_path: Path to the dataset for evaluation.

    Returns:
    - metrics: Dictionary containing accuracy, F1 score, precision, recall, AUC-ROC averaged over all classes.
    """

    # Load the dataset
    test_data = load_test_data(dataset_path)  # Assuming this function can take a path

    # Load the model
    model = load_model(model_path)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

    # Convert lists to numpy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.concatenate(all_probs)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean() * 100
    f1 = calculate_metrics(all_labels, all_predictions, metric='f1')
    precision = calculate_metrics(all_labels, all_predictions, metric='precision')
    recall = calculate_metrics(all_labels, all_predictions, metric='recall')
    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc
    }

    return metrics

# Example usage
# model_path = 'path/to/your/classifier_model.pth'
# dataset_path = 'path/to/your/test_dataset'
# metrics = evaluate_classifier(model_path, dataset_path)
# print(metrics)
