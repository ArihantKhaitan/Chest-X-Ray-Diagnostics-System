import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

IMG_SIZE = (224, 224)
NUM_CLASSES = 4  # NORMAL, PNEUMONIA, TB, LUNG_CANCER

# Custom CNN
def build_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Transfer learning base function
def build_transfer_model(base_model):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# VGG16, ResNet50, EfficientNet
def build_vgg16():
    base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    return build_transfer_model(base)

def build_resnet50():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    return build_transfer_model(base)

def build_efficientnet():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    return build_transfer_model(base)

# Fine-tune function
def fine_tune(model, layers_to_unfreeze=5):
    if len(model.layers) > 1 and hasattr(model.layers[0], 'layers'):
        base_model = model.layers[0]
        base_model.trainable = True
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model