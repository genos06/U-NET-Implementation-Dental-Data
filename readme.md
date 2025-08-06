# Solution Approach

## Attempted Methods & Architectures

- **Baseline UNet**: Implemented a standard UNet architecture for semantic segmentation.  
- **Custom Data Preprocessing**: Explored various mask generation strategies from polygon annotations.
- **Augmentation**: Tried basic augmentations (horizontal/vertical flips, resizing).
- **Loss Functions**: Used BCEWithLogitsLoss for binary segmentation.
- **Learning Rate Schedulers**: Tried various learning rate schedulers (StepLR, ReduceLROnPlateau, CosineAnnealingLR) and different learning rates, but did not have enough computation time on Colab to evaluate these approaches reasonably.

## Final Chosen Approach

### Preprocessing
- Masks are generated from polygon annotations using PIL's `ImageDraw`.
- Images and masks are resized to (160, 240) using Torchvision.
- Augmentations: random horizontal/vertical flips.

### Model Architecture
- Standard UNet with configurable input/output channels.
- Encoder-decoder structure with skip connections.
- Final output is a single-channel mask (binary segmentation).

### Training Strategy
- Adam optimizer, learning rate 1e-4.
- BCEWithLogitsLoss.
- Batch size: 32.
- Training for 100 epochs. (got only 20 due to computation restrictions)

### Post-processing
- Output masks are thresholded for visualization.

## Future Improvements

- Try IOU loss for better segmentation metrics and loss function.
- Experiment with deeper UNet.
- Hyperparameter tuning (learning rate, batch size, scheduler).
- Ensemble multiple models for robust predictions.
- Further explore learning rate schedulers and different learning rates if more compute resources are available.

# Reproducibility Instructions

## Requirements

Use the `requirements.txt` for installation of required dependancies.


## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare the dataset:
   - Place your .tar file in the same directory as expected by the code.

## Model Weights

Model weights are not included in the repository due to the GitHub file size limit (100MB).  
You can download the trained model weights from the following Google Drive link:

[Download Model Weights](https://drive.google.com/file/d/1R1Wp-7ZiQBOqmHvsJ061_9uUwg_tJJlK/view?usp=sharing)

After downloading, place the weights file in the expected directory as referenced by the code/notebooks.

## Training

Run the training notebook:
```
jupyter notebook training.ipynb
```

## Inference

Run the inference notebook:
```
jupyter notebook results.ipynb
```

# Codebase

- `preprocessing.ipynb`: Data exploration and mask generation.
- `training.ipynb`: Model training pipeline.
- `results.ipynb`: Inference and visualization.
