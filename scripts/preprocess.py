import numpy as np

def process_pixels(pixels):
    
    features = []
    for pixel in pixels:
        arr = np.array(pixel.split(), dtype=np.uint8)
        features.append(arr)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)

    return features / 255