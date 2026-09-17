from tensorflow.keras.applications import MobileNetV2

def create_mobilenet(_):
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    mobilenet.trainable = True

    for layer in mobilenet.layers[:-40]:
        layer.trainable = False

    return mobilenet
