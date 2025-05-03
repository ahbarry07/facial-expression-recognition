from PIL import Image
import numpy as np
import cv2


def process_pixels(pixels):
    """
    Normalize and reshape a list of pixel strings into a CNN-compatible array.

    This function processes a list of pixel data (typically as space-separated strings)
    by converting each string into a numpy array of uint8 values, reshaping it into
    48x48 grayscale images with a channel dimension, and normalizing the pixel values
    to the range [0, 1] for CNN input.

    Args:
        pixels (list): List of strings where each string represents pixel values
                       (e.g., from a CSV file like FER-2013 dataset).

    Returns:
        numpy.ndarray: Preprocessed image array with shape (n, 48, 48, 1) where n is
                       the number of images, and values normalized to [0, 1].

    Raises:
        ValueError: If the pixel strings cannot be converted to valid numpy arrays
                    or if the resulting shape is incompatible with (48, 48, 1).
        TypeError: If the input `pixels` is not a list or contains non-string elements.

    Notes:
        - Assumes each string in `pixels` can be split into 2304 values (48*48) representing
          a flattened 48x48 grayscale image.
        - The function uses `numpy` for array operations and requires it to be imported.
    """

    features = []
    for pixel in pixels:
        arr = np.array(pixel.split(), dtype=np.uint8)
        features.append(arr)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)

    return features / 255



def preprocess_frame(face_img, img_size):
    """
    Preprocess a face image to match CNN input (grayscale, resized to specified size).

    This function converts a BGR image frame to grayscale, resizes it to the specified
    dimensions, normalizes pixel values to the range [0, 1], and adds batch and channel
    dimensions for CNN input.

    Args:
        face_img (numpy.ndarray): Input image frame in BGR format with shape (height, width, 3).
        img_size (int): Target size for resizing (e.g., 48 for 48x48 images).

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, img_size, img_size, 1)
                       and values normalized to [0, 1].

    Raises:
        ValueError: If the input `face_img` is None, has invalid dimensions, or cannot be
                    converted to grayscale.
        cv2.error: If the OpenCV operations (e.g., `cv2.cvtColor`, `cv2.resize`) fail.

    Notes:
        - Requires `cv2` (OpenCV) to be imported for image processing.
        - The function assumes the input is a valid BGR image from a video frame or similar source.
        - Normalization divides by 255.0 to match typical CNN input requirements.
    """

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    return normalized.reshape(1, img_size, img_size, 1)



def convert_image(img_path):
    """
    Load and preprocess an image for CNN input.

    Reads an image from the specified path, resizes it to 48x48 pixels, converts it to grayscale,
    normalizes pixel values to the range [0, 1], and formats it for CNN input with batch and channel
    dimensions. This function is optimized for emotion recognition models expecting 48x48 grayscale
    images.

    Args:
        img_path (str): Path to the input image file (e.g., '.png', '.jpg', '.jpeg').

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 48, 48, 1) and values normalized
                       to the range [0, 1], suitable for CNN input.

    Raises:
        FileNotFoundError: If the image file at `img_path` does not exist.
        ValueError: If the image file cannot be identified or opened (e.g., corrupted file or
                    unsupported format).
        TypeError: If `img_path` is not a string.

    Notes:
        - The image is resized to a fixed 48x48 resolution, which may lead to loss of detail
          for larger images; use with caution for high-resolution inputs.
        - Uses PIL (Python Imaging Library) for image loading and conversion to grayscale.
        - Normalization is performed by dividing by 255.0 to match typical CNN input requirements.
        - Requires `PIL.Image`, `numpy`, and their dependencies to be imported.
        - Performance may vary with large datasets due to sequential image processing.

    Examples:
        >>> img = convert_image("path/to/image.png")
        >>> print(img.shape)  # Output: (1, 48, 48, 1)
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
