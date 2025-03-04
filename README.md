# Chest X-Ray Pneumonia Classification using Deep Learning

This project implements a deep learning solution for pneumonia detection from chest X-ray images. It features GPU-optimized neural network models with various architectures, including custom CNNs and transfer learning approaches.

## Features

- ✅ **Multiple model architectures**: Custom CNN, optimized CNN, and transfer learning models (EfficientNetB0, MobileNetV2, VGG16)
- ✅ **Advanced GPU optimization**: Mixed precision training, XLA compilation, memory growth settings
- ✅ **Efficient data pipeline**: Optimized tf.data API implementation with prefetching and caching
- ✅ **Comprehensive data augmentation**: Includes flipping, cropping, brightness/contrast adjustments
- ✅ **K-fold cross-validation**: Statistical validation with detailed performance metrics
- ✅ **Visualization tools**: Training progress and cross-validation results visualization

## Dataset

The project expects chest X-ray images organized in the following directory structure:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/ (optional)
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/ (optional)
    ├── NORMAL/
    └── PNEUMONIA/
```

You can use the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) or any similarly structured dataset.

## Requirements

```
tensorflow>=2.4.0
numpy
pandas
matplotlib
scikit-learn
tqdm
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/chest-xray-pneumonia-classification.git
cd chest-xray-pneumonia-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the project directory.

## Usage

### Training

To train the model with cross-validation:

```python
from model import prepare_cv_data, perform_optimized_cross_validation, create_gpu_optimized_model

# Load data
all_files, all_labels = prepare_cv_data()

# Train with cross-validation
cv_results, cv_df, final_model = perform_optimized_cross_validation(
    n_splits=5,
    epochs=20,
    batch_size=64,
    model_func=create_gpu_optimized_model
)

# Save the trained model
final_model.save('pneumonia_model_gpu_optimized.h5')
```

### Choosing Different Models

The project includes several model architectures:

```python
# Basic CNN model
cv_results, cv_df, model = perform_optimized_cross_validation(
    model_func=create_model
)

# GPU-optimized CNN
cv_results, cv_df, model = perform_optimized_cross_validation(
    model_func=create_gpu_optimized_model
)

# Transfer learning with EfficientNetB0
cv_results, cv_df, model = perform_optimized_cross_validation(
    model_func=lambda: create_optimized_transfer_learning_model('EfficientNetB0')
)

# Transfer learning with MobileNetV2
cv_results, cv_df, model = perform_optimized_cross_validation(
    model_func=lambda: create_optimized_transfer_learning_model('MobileNetV2')
)

# Transfer learning with VGG16
cv_results, cv_df, model = perform_optimized_cross_validation(
    model_func=lambda: create_optimized_transfer_learning_model('VGG16')
)
```

## GPU Optimization

This project implements several GPU optimization techniques:

1. **Mixed Precision Training**: Uses float16 for faster computation where appropriate
2. **XLA (Accelerated Linear Algebra) Compilation**: Optimizes TensorFlow computation graphs
3. **Memory Growth**: Prevents TensorFlow from allocating all GPU memory at once
4. **Optimized tf.data Pipeline**: Uses prefetching, caching, and parallel processing
5. **Batch Size Tuning**: Configurable batch sizes to maximize GPU utilization

## Model Architectures

### Custom CNN
A standard convolutional neural network with three convolutional blocks followed by fully connected layers.

### GPU-Optimized CNN
An enhanced CNN with batch normalization, dropout, and a more efficient global average pooling layer, designed to better utilize GPU parallelism.

### Transfer Learning Models
Pre-trained models (EfficientNetB0, MobileNetV2, VGG16) with customized classification heads for pneumonia detection.

## Performance Metrics

The training process tracks and reports:
- Accuracy
- Loss
- Precision
- Recall
- AUC (Area Under the ROC Curve)

Results are saved in CSV format and visualized with matplotlib.

## Project Structure

```
.
├── model.py          # Main implementation file
├── README.md         # This file
├── requirements.txt  # Dependencies
└── logs/             # Training logs and visualizations
    └── pneumonia_cv_gpu_optimized/
        ├── fold_1/
        ├── fold_2/
        └── ...
```

## Notes

- The code contains Norwegian comments which can be translated if needed.
- Model performance may vary based on GPU hardware and available memory.
- Adjust batch size based on your GPU memory capacity for optimal performance.

## License

[MIT License](LICENSE)