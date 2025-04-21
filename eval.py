import os
import sys
import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py eval_ds")
        sys.exit(1)

    # Evaluation dataset directory
    EVAL_DIR = sys.argv[1] if len(sys.argv) > 1 else "eval_ds"  # Default to "eval_ds"


    # Path to the saved model
    MODEL_PATH = "model.pth"

    # Class names
    CLASS_NAMES = ['safe', 'violent']

    # Set device to GPU if available, else fallback to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    print("Loading model...")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()

    # Data transformations
    eval_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transforms)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, shuffle=False, num_workers=0  # Avoid multiprocessing issues
    )

    # Evaluate
    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    conf_mat = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(conf_mat)

    class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {CLASS_NAMES[i]}: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = conf_mat.max() / 2
    for i, j in np.ndindex(conf_mat.shape):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    main()


# python eval.py eval_ds