import matplotlib.pyplot as plt
import numpy as np
import cv2

def summarize_diagnostics(history):
    """
    Plots and saves the training and validation accuracy and loss curves.

    This function generates a two-panel plot to visualize the training and validation
    accuracy and loss over epochs, aiding in the assessment of model performance and
    detection of overfitting. The plots are saved as a PNG file and displayed interactively.

    Args:
        history: History object returned by a Keras model's `fit` method, containing
                 training metrics (e.g., 'accuracy', 'val_accuracy', 'loss', 'val_loss').

    Returns:
        None: The function saves the plot to a file and displays it but does not return a value.

    Raises:
        KeyError: If the `history` object lacks required keys ('accuracy', 'val_accuracy',
                  'loss', 'val_loss') due to incomplete training or invalid data.
        ValueError: If the plot fails to save due to an invalid file path or permissions.
        ImportError: If `matplotlib.pyplot` is not available.

    Notes:
        - The plot consists of two subplots: one for accuracy (left) and one for loss (right).
        - The figure size is set to 12x4 inches for readability.
        - Labels are in English ('Epochs', 'Accuracy', 'Loss', 'Evolution of Accuracy',
          'Evolution of Loss') to match the updated code.
        - The plot is saved as './results/learn_and_loss_curves.png' and displayed using `plt.show()`.
        - Requires `matplotlib.pyplot` to be imported as `plt` in the global scope.
    """

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Evolution of Accuracy")
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Evolution of Loss")

    plt.savefig("./results/learn_and_loss_curves.png")
    plt.show()



def save_and_compare_images(original_image, modified_image, original_emotion, original_prob, modified_emotion, modified_prob, output_dir):
    """
    Save original and modified images and create a comparison plot.

    Args:
        original_image: Original image array (48, 48).
        modified_image: Modified image array (48, 48).
        original_emotion (str): Predicted emotion for original image.
        original_prob (float): Probability for original prediction.
        modified_emotion (str): Predicted emotion for modified image.
        modified_prob (float): Probability for modified prediction.
        output_dir (str): Directory to save the images.
    """

    original_save = (original_image * 255).astype(np.uint8)
    modified_save = (modified_image * 255).astype(np.uint8)

    cv2.imwrite(f'{output_dir}/original_happy_image.png', original_save)
    cv2.imwrite(f'{output_dir}/modified_sad_image.png', modified_save)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {original_emotion} ({original_prob:.2f})")
    plt.imshow(original_save, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f"Modified: {modified_emotion} ({modified_prob:.2f})")
    plt.imshow(modified_save, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png')
    plt.close()