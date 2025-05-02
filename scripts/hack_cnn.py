import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Constants
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48  # Expected input size for the CNN (48x48)
MODEL_PATH = './model/final_emotion_model.keras'
OUTPUT_DIR = './results'
HAPPY_IDX = EMOTION_LABELS.index('Happy')  # Index 3
SAD_IDX = EMOTION_LABELS.index('Sad')      # Index 4
MIN_HAPPY_PROB = 0.9  # Minimum probability for 'Happy' prediction
ALPHA = 0.001         # Step size for adversarial attack
MAX_ITERATIONS = 100  # Maximum iterations for adversarial attack



def convert_image(img_path):
    """
    Load and preprocess an image for CNN input.

    Reads an image from the specified path, resizes it to 48x48 pixels, converts it to grayscale,
    normalizes pixel values to the range [0, 1], and formats it for CNN input with batch and channel
    dimensions.

    Args:
        img_path (str): Path to the input image file.

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 48, 48, 1).

    Raises:
        FileNotFoundError: If the image file at `img_path` does not exist.
        ValueError: If the image file cannot be identified or opened (e.g., corrupted file).
    """

    try:
        img = Image.open(img_path).resize((48, 48))  # Resize to 48x48
        img = img.convert('L')  # Convert to grayscale

        img = np.array(img)
        img = img.astype("float32") / 255.0  # Normalize to [0, 1]

        img = np.expand_dims(img, axis=-1)  # Add channel: (48, 48, 1)
        img = np.expand_dims(img, axis=0)   # Add batch: (1, 48, 48, 1)

        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at: {img_path}")
    except Image.UnidentifiedImageError:
        raise ValueError(f"Cannot identify or open image file at: {img_path}")



def predict_emotion(model, img_processed, emotion_labels):
    """
    Predict the emotion of a preprocessed image using the CNN model.

    Args:
        model: Loaded TensorFlow/Keras model for emotion prediction.
        image: Preprocessed image with shape (1, 48, 48, 1) or (48, 48).
        emotion_labels (list): List of emotion labels.

    Returns:
        tuple: Predicted emotion (str) and probability (float, percentage).
    """
     
    img_processed = img_processed.reshape(1, 48, 48, 1)
    prediction = model.predict(img_processed, verbose=0)
    return emotion_labels[np.argmax(prediction)], np.max(prediction) * 100




def adversarial_attack(image, model, target_class_idx, emotion_labels, max_iterations=MAX_ITERATIONS, alpha=ALPHA):
    """
    Perform an adversarial attack to change the CNN's prediction to a target emotion.

    Args:
        image: Preprocessed image with shape (1, 48, 48, 1).
        model: Loaded TensorFlow/Keras model for emotion prediction.
        target_class_idx (int): Index of the target emotion label.
        emotion_labels (list): List of emotion labels.
        max_iterations (int): Maximum number of iterations for the attack.
        alpha (float): Step size for gradient-based perturbation.

    Returns:
        numpy.ndarray: Adversarial image with shape (48, 48).
    """

    # Convert the image to a TensorFlow tensor and ensure it requires gradients
    adv_image_tensor = tf.Variable(image, dtype=tf.float32)
    target = tf.expand_dims(tf.one_hot(target_class_idx, len(emotion_labels)), axis=0)

    for iteration in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image_tensor)
            predictions = model(adv_image_tensor)
            loss = tf.keras.losses.categorical_crossentropy(target, predictions)

        # Compute the gradient of the loss with respect to the imagE
        gradient = tape.gradient(loss, adv_image_tensor)
        signed_grad = tf.sign(gradient)

        # Apply the perturbation (small step in the direction of the gradient)
        adv_image_tensor.assign_add(-alpha * signed_grad)

        # Clip the image to ensure pixel values remain in [0, 1]
        adv_image_tensor.assign(tf.clip_by_value(adv_image_tensor, 0.0, 1.0))

        # Check the current prediction
        adv_image_np = adv_image_tensor.numpy().squeeze(axis=0).squeeze(axis=-1)
        current_pred, current_prob = predict_emotion(model, adv_image_np, emotion_labels)
        print(f"Iteration {iteration+1}: Predicted Emotion: {current_pred}, Probability: {current_prob:.4f}")

        if current_pred == emotion_labels[target_class_idx]:
            break

    return adv_image_tensor.numpy().squeeze(axis=0).squeeze(axis=-1)



def save_and_compare_images(original_image, modified_image, original_emotion, original_prob, modified_emotion, modified_prob, output_dir=OUTPUT_DIR):
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


def main():
    """Main function to perform adversarial attack on an image."""

    model = tf.keras.models.load_model(MODEL_PATH)
    image_path = f"{OUTPUT_DIR}/happy.png"
    
    # Preprocess and predict original image
    processed_image = convert_image(image_path)
    original_emotion, original_prob = predict_emotion(model, processed_image, EMOTION_LABELS)
    print(f"Original Prediction: {original_emotion}, Probability: {original_prob:.4f}")

    # Validate original prediction
    if original_emotion != 'Happy' or original_prob / 100 < MIN_HAPPY_PROB:
        print("Please select an image that the CNN predicts as 'Happy' with >90% probability.")
        return

    # Perform adversarial attack
    print("\nPerforming adversarial attack to change prediction to 'Sad'...")
    modified_image = adversarial_attack(processed_image, model, SAD_IDX, EMOTION_LABELS)

    # Predict modified image
    modified_emotion, modified_prob = predict_emotion(model, modified_image, EMOTION_LABELS)
    print(f"\nModified Prediction: {modified_emotion}, Probability: {modified_prob:.4f}")

    # Save and compare images
    original_image_np = processed_image.squeeze(axis=0).squeeze(axis=-1)
    save_and_compare_images(original_image_np, modified_image, original_emotion, original_prob,
                           modified_emotion, modified_prob)

    print("\nImages saved:")
    print(f"- Original: {OUTPUT_DIR}/original_happy_image.png")
    print(f"- Modified: {OUTPUT_DIR}/modified_sad_image.png")
    print(f"- Comparison: {OUTPUT_DIR}/comparison.png")



if __name__ == "__main__":
    main()