# Dataset Structure Guide

Your dataset should be organized in the following structure:

```
dataset/
├── train/
│   ├── colon_adenocarcinoma/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── colon_benign/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── colon_malignant/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── normal_colon/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/  (optional)
    ├── colon_adenocarcinoma/
    ├── colon_benign/
    ├── colon_malignant/
    └── normal_colon/
```

## Training Instructions

1. Prepare your dataset:
   - Place your images in the corresponding class folders under `dataset/train/`
   - Each image should be in JPG/JPEG/PNG format
   - Images will be automatically resized to 256x256 pixels

2. Install required packages:
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn
   ```

3. Run the training:
   ```bash
   python train_model.py
   ```

4. The script will:
   - Split training data into train/validation sets (80%/20%)
   - Train the model with data augmentation
   - Save the best model to `model/colon_diseases.h5`
   - Generate training plots (accuracy/loss curves)
   - If test set provided, generate confusion matrix and classification report

5. Monitor the output:
   - Watch the training progress in the console
   - Check the generated plots in the project directory
   - The best model will be saved automatically

## Model Architecture

The CNN architecture includes:
- 3 convolutional blocks with increasing filters (32→64→128)
- Batch normalization for better training stability
- Dropout layers to prevent overfitting
- Dense layers for final classification

## Data Augmentation

The training includes the following augmentations:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Shear transformation
- Zoom (±20%)
- Horizontal flips

## Early Stopping

Training will automatically stop if validation loss doesn't improve for 10 epochs.
Learning rate is reduced if no improvement is seen for 5 epochs.

## Outputs

The training will generate:
- Trained model: `model/colon_diseases.h5`
- Training history plot: `training_history.png`
- Confusion matrix (if test set available): `confusion_matrix.png`
- Detailed classification report in the console

## Customization

You can modify these parameters in `train_model.py`:
- `IMAGE_SIZE`: Input image size (default: 256)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `EPOCHS`: Maximum number of training epochs (default: 50)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)