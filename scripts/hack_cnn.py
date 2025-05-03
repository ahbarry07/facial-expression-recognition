import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

from predict_live_stream import predict_emotion
from plot import save_and_compare_images
from preprocess import convert_image


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



def adversarial_attack(image, model, target_class_idx, emotion_labels, max_iterations=MAX_ITERATIONS, alpha=ALPHA):
    """
    Perform an adversarial attack to change the CNN's prediction to a target emotion.

    This function implements a gradient-based adversarial attack to subtly perturb an input
    image such that the trained CNN model misclassifies it as a specified target emotion.
    The attack iteratively adjusts the image pixels using the sign of the gradient of the
    loss with respect to the input, clipping values to maintain validity, until the target
    emotion is achieved or the maximum iterations are reached.

    Args:
        image: Preprocessed image with shape (1, 48, 48, 1) and values in range [0, 1].
        model: Loaded TensorFlow/Keras model for emotion prediction.
        target_class_idx (int): Index of the target emotion label in `emotion_labels`.
        emotion_labels (list): List of emotion labels (e.g., ['Angry', 'Disgust', ...]).
        max_iterations (int, optional): Maximum number of iterations for the attack.
            Defaults to the global constant `MAX_ITERATIONS`.
        alpha (float, optional): Step size for each gradient-based perturbation.
            Defaults to the global constant `ALPHA`.

    Returns:
        numpy.ndarray: Adversarial image with shape (48, 48) and values in range [0, 1].

    Raises:
        ValueError: If the input image has an invalid shape, if `target_class_idx` is out of
            range of `emotion_labels`, or if `max_iterations` or `alpha` are negative.
        tf.errors.InvalidArgumentError: If the model fails to process the input due to shape
            mismatch or invalid tensor operations.
        AttributeError: If `predict_emotion` or global constants (`MAX_ITERATIONS`, `ALPHA`)
            are not defined.

    Notes:
        - The function uses TensorFlow's GradientTape to compute gradients and applies
          perturbations in the direction that minimizes the loss for the target class.
        - Pixel values are clipped to [0, 1] to ensure the image remains valid.
        - The attack stops early if the target emotion is predicted, reducing unnecessary
          iterations.
        - The function relies on the global function `predict_emotion` for checking predictions
          and assumes it is defined with the `isFrame=False` parameter.
        - Performance depends on the initial image quality and the model's sensitivity to
          perturbations.
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
        current_pred, current_prob = predict_emotion(model, adv_image_np, emotion_labels, isFrame=False)
        print(f"Iteration {iteration+1}: Predicted Emotion: {current_pred}, Probability: {current_prob:.4f}")

        if current_pred == emotion_labels[target_class_idx]:
            break

    return adv_image_tensor.numpy().squeeze(axis=0).squeeze(axis=-1)


def main():
    """Main function to perform adversarial attack on an image."""

    model = tf.keras.models.load_model(MODEL_PATH)
    image_path = f"{OUTPUT_DIR}/happy.png"
    
    # Preprocess and predict original image
    processed_image = convert_image(image_path)
    original_emotion, original_prob = predict_emotion(model, processed_image, EMOTION_LABELS, isFrame=False)
    print(f"Original Prediction: {original_emotion}, Probability: {original_prob:.4f}")

    # Validate original prediction
    if original_emotion != 'Happy' or original_prob  < MIN_HAPPY_PROB:
        print("Please select an image that the CNN predicts as 'Happy' with >90% probability.")
        return

    # Perform adversarial attack
    print("\nPerforming adversarial attack to change prediction to 'Sad'...")
    modified_image = adversarial_attack(processed_image, model, SAD_IDX, EMOTION_LABELS)

    # Predict modified image
    modified_emotion, modified_prob = predict_emotion(model, modified_image, EMOTION_LABELS, isFrame=False)
    print(f"\nModified Prediction: {modified_emotion}, Probability: {modified_prob:.4f}")

    # Save and compare images
    original_image_np = processed_image.squeeze(axis=0).squeeze(axis=-1)
    save_and_compare_images(original_image_np, modified_image, original_emotion, original_prob,
                           modified_emotion, modified_prob, output_dir=OUTPUT_DIR)

    print("\nImages saved:")
    print(f"- Original: {OUTPUT_DIR}/original_happy_image.png")
    print(f"- Modified: {OUTPUT_DIR}/modified_sad_image.png")
    print(f"- Comparison: {OUTPUT_DIR}/comparison.png")



if __name__ == "__main__":
    main()