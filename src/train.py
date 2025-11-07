import os
from data_prep import train_generator, validation_generator, test_generator  # Import only generators
from models import build_custom_cnn, build_vgg16, build_resnet50, build_efficientnet, fine_tune

os.makedirs('saved_models', exist_ok=True)

def train_model(model_builder, name, epochs=20):
    model = model_builder()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    model = fine_tune(model)
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"{name} Test Accuracy: {test_accuracy:.4f}")
    model.save(f'saved_models\\{name}.h5')
    return model, history, history_fine

# Retrain all models
custom_model, custom_hist, custom_hist_fine = train_model(build_custom_cnn, 'custom_cnn')
vgg_model, vgg_hist, vgg_hist_fine = train_model(build_vgg16, 'vgg16')
resnet_model, resnet_hist, resnet_hist_fine = train_model(build_resnet50, 'resnet50')
effnet_model, effnet_hist, effnet_hist_fine = train_model(build_efficientnet, 'efficientnet')