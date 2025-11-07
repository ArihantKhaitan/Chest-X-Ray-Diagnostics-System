import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
pneumonia_dir = 'subsampled_chest_xray'
tb_dir = 'TB_dataset'
lung_cancer_dir = 'LungCancer_dataset'
multi_dataset_dir = 'multi_disease_chest_xray'
classes = ['NORMAL', 'PNEUMONIA', 'TB', 'LUNG_CANCER']

# Clear previous multi-disease dataset to start fresh
if os.path.exists(multi_dataset_dir):
    import shutil
    shutil.rmtree(multi_dataset_dir)
    print(f"Cleared existing {multi_dataset_dir}")

# Create directory structure
for split in ['train', 'val', 'test']:
    for class_name in classes:
        os.makedirs(os.path.join(multi_dataset_dir, split, class_name), exist_ok=True)

# Subsample function
def subsample_and_copy(src_dir, dest_dir, num_samples):
    if not os.path.exists(src_dir):
        print(f"Warning: Source directory {src_dir} does not exist, skipping.")
        return
    images = [f for f in os.listdir(src_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    if len(images) == 0:
        print(f"Warning: No images found in {src_dir}")
        return
    if len(images) <= num_samples:
        selected_images = images
    else:
        selected_images = random.sample(images, num_samples)
    copied = 0
    for img in selected_images:
        try:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))
            copied += 1
        except Exception as e:
            print(f"Error copying {img}: {e}")
    print(f"Copied {copied}/{len(selected_images)} images from {src_dir} to {dest_dir}")

# Subsample Pneumonia
print("Subsampling Pneumonia data...")
subsample_and_copy(os.path.join(pneumonia_dir, 'train', 'NORMAL'), os.path.join(multi_dataset_dir, 'train', 'NORMAL'), 70)
subsample_and_copy(os.path.join(pneumonia_dir, 'train', 'PNEUMONIA'), os.path.join(multi_dataset_dir, 'train', 'PNEUMONIA'), 70)
subsample_and_copy(os.path.join(pneumonia_dir, 'test', 'NORMAL'), os.path.join(multi_dataset_dir, 'test', 'NORMAL'), 15)
subsample_and_copy(os.path.join(pneumonia_dir, 'test', 'PNEUMONIA'), os.path.join(multi_dataset_dir, 'test', 'PNEUMONIA'), 15)
subsample_and_copy(os.path.join(pneumonia_dir, 'val', 'NORMAL'), os.path.join(multi_dataset_dir, 'val', 'NORMAL'), 15)
subsample_and_copy(os.path.join(pneumonia_dir, 'val', 'PNEUMONIA'), os.path.join(multi_dataset_dir, 'val', 'PNEUMONIA'), 15)

# Subsample TB
print("Subsampling TB data...")
subsample_and_copy(os.path.join(tb_dir, 'Normal'), os.path.join(multi_dataset_dir, 'train', 'NORMAL'), 70)
subsample_and_copy(os.path.join(tb_dir, 'Tuberculosis'), os.path.join(multi_dataset_dir, 'train', 'TB'), 70)
subsample_and_copy(os.path.join(tb_dir, 'Normal'), os.path.join(multi_dataset_dir, 'test', 'NORMAL'), 15)
subsample_and_copy(os.path.join(tb_dir, 'Tuberculosis'), os.path.join(multi_dataset_dir, 'test', 'TB'), 15)
subsample_and_copy(os.path.join(tb_dir, 'Normal'), os.path.join(multi_dataset_dir, 'val', 'NORMAL'), 15)
subsample_and_copy(os.path.join(tb_dir, 'Tuberculosis'), os.path.join(multi_dataset_dir, 'val', 'TB'), 15)

# Subsample Lung Cancer
print("Subsampling Lung Cancer data...")
lung_iq_folder = "The IQ-OTHNCCD lung cancer dataset"
lung_nested_folder = "The IQ-OTHNCCD lung cancer dataset"  # Nested folder
lung_iq_path = os.path.join(lung_cancer_dir, lung_iq_folder, lung_nested_folder)
if os.path.exists(lung_iq_path):
    lung_subfolders = [f for f in os.listdir(lung_iq_path) if os.path.isdir(os.path.join(lung_iq_path, f))]
    print(f"Found Lung Cancer subfolders: {lung_subfolders}")
    for subfolder in lung_subfolders:
        subfolder_path = os.path.join(lung_iq_path, subfolder)
        if subfolder.lower() == 'normal cases':
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'train', 'NORMAL'), 70)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'test', 'NORMAL'), 15)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'val', 'NORMAL'), 15)
        elif subfolder.lower() == 'malignant cases':
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'train', 'LUNG_CANCER'), 70)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'test', 'LUNG_CANCER'), 15)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'val', 'LUNG_CANCER'), 15)
        elif subfolder.lower() == 'bengin cases':
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'train', 'NORMAL'), 70)  # Treat as benign/normal
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'test', 'NORMAL'), 15)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'val', 'NORMAL'), 15)
        else:
            print(f"Unknown Lung Cancer subfolder: {subfolder}, mapping to LUNG_CANCER")
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'train', 'LUNG_CANCER'), 70)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'test', 'LUNG_CANCER'), 15)
            subsample_and_copy(subfolder_path, os.path.join(multi_dataset_dir, 'val', 'LUNG_CANCER'), 15)
else:
    print(f"Warning: {lung_iq_path} not found. Skipping Lung Cancer data.")

print("Multi-disease dataset preparation completed.")

# Data generators for multi-class
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(multi_dataset_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(multi_dataset_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(multi_dataset_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Train images: {train_generator.samples}, Val images: {validation_generator.samples}, Test images: {test_generator.samples}")
print(f"Classes: {train_generator.class_indices}")