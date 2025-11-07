import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from data_prep import test_generator
from sklearn.preprocessing import label_binarize

# Load models
models_list = [
    load_model('saved_models/custom_cnn.h5'),
    load_model('saved_models/vgg16.h5'),
    load_model('saved_models/resnet50.h5'),
    load_model('saved_models/efficientnet.h5')
]
model_names = ['Custom CNN', 'VGG16', 'ResNet50', 'EfficientNet']

# Get true labels and class names
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())  # ['LUNG_CANCER', 'NORMAL', 'PNEUMONIA', 'TB']
n_classes = len(class_names)

# Evaluate single model
def evaluate_model(model, name):
    # Evaluate accuracy
    loss, acc = model.evaluate(test_generator)
    print(f'{name} - Test Accuracy: {acc:.4f}')

    # Predictions
    y_pred = model.predict(test_generator)
    y_pred_class = np.argmax(y_pred, axis=1)  # Multi-class prediction
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # Classification Report
    print(f'{name} Classification Report:')
    print(classification_report(y_true, y_pred_class, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # ROC Curve (One-vs-Rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = roc_auc_score(y_true_binarized[:, i], y_pred[:, i])
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Evaluate all models
for model, name in zip(models_list, model_names):
    evaluate_model(model, name)

# Ensemble evaluation
def ensemble_eval(models_list):
    steps = len(test_generator)
    test_generator.reset()
    y_preds = []
    for _ in range(steps):
        x, y = next(test_generator)
        pred = np.mean([m.predict(x) for m in models_list], axis=0)
        y_preds.extend(pred)
    y_pred_class = np.argmax(np.array(y_preds), axis=1)[:len(test_generator.classes)]
    y_true = test_generator.classes
    print('Ensemble Classification Report:')
    print(classification_report(y_true, y_pred_class, target_names=class_names, zero_division=0))

ensemble_eval(models_list)