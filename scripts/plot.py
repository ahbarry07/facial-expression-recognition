import matplotlib.pyplot as plt


def summarize_diagnostics(history):
    """
    Plots and saves the training and validation accuracy and loss curves.
    Helps to visualize model performance and detect overfitting.
    """
    
    plt.figure(figsize=(12, 4))

    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Épochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Évolution de la précision")
    
    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Épochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Évolution de la perte")

    
    plt.savefig("./results/learn_and_loss_curves.png")

    plt.show()