Polyp Segmentation with PraNet on Kvasir-SEG Dataset

This repository contains the implementation of a polyp segmentation model using the PraNet architecture on the Kvasir-SEG dataset. The project aims to achieve high performance in medical image segmentation, maximizing metrics such as accuracy, precision, recall, F1 score, AUC, Dice coefficient, and IoU, while addressing overfitting through advanced techniques.

Project Overview

Polyps are abnormal growths in the colon that can lead to colorectal cancer. Accurate segmentation of polyps in endoscopic images is critical for early diagnosis and treatment. This project leverages the PraNet model, a deep learning architecture designed for polyp segmentation, trained on the Kvasir-SEG dataset to achieve state-of-the-art performance.

Objectives





Maximize segmentation performance across multiple metrics: accuracy, precision, recall, F1 score, AUC, Dice coefficient, and IoU.



Mitigate overfitting using data augmentation, regularization, and early stopping.



Provide a reproducible implementation for running on Kaggle with clear instructions.

Dataset

The Kvasir-SEG dataset contains 1000 endoscopic images of polyps with corresponding binary masks. Each image-mask pair is used for training, validation, and testing, split as follows:





Training: 80% (800 images)



Validation: 10% (100 images)



Test: 10% (100 images)

Source: Kvasir-SEG Dataset on Kaggle

Methodology

Model Architecture

The PraNet model uses a ResNet50 backbone with Reverse Attention (RA) modules to focus on polyp regions. Key components:





Backbone: ResNet50 pre-trained on ImageNet.



RA Modules: Three RA blocks (2048→512, 1024→256, 512→64) to refine features.



Output: Single-channel segmentation map upsampled to 352x352.

Optimizations

To maximize metrics and address overfitting, the following techniques were applied:





Data Augmentation:





Horizontal/vertical flips, random 90° rotations.



Color jitter (brightness, contrast, saturation, hue).



Elastic transforms and Gaussian noise.



Loss Function: Combined BCE (with custom label smoothing) + Dice loss (50:50 weight) for balanced pixel-wise and region-based optimization.



Regularization:





Dropout (20%) in RA modules.



Weight decay (2e-2) in AdamW optimizer.



Learning Rate Scheduling: Cosine annealing (T_max=30, eta_min=1e-6).



Early Stopping: Patience of 10 epochs based on validation loss.



Test-Time Augmentation (TTA): Averaged predictions from original, flipped, and rotated images.



Threshold Optimization: Selected optimal threshold (0.60) to maximize F1 score on validation set.



Mixed Precision Training: Used torch.amp for faster training on Kaggle’s P100 GPU.



Efficient Data Loading: num_workers=2, pin_memory=True.

Training





Epochs: Up to 30 (with early stopping).



Batch Size: 8.



Optimizer: AdamW (lr=1e-4).



Device: NVIDIA P100 GPU (Kaggle).

Results

The model was evaluated on the test set (100 images) with an optimal threshold of 0.60. The final test metrics are:







Metric



Value





Accuracy



0.9691





Precision



0.8653





Recall



0.9403





F1 Score



0.8990





AUC



0.9925





Dice Coefficient



0.8990





IoU



0.8215





True Positives



134,205





True Negatives



808,019





False Positives



20,937





False Negatives



8,244

Training Progress





Training Loss: Decreased from 0.5355 (epoch 1) to 0.2096 (epoch 30).



Validation Loss: Decreased from 0.4260 (epoch 1) to 0.2376 (epoch 26, best), indicating reduced overfitting.



Validation Metrics (epoch 26, best):





Dice: 0.8910



IoU: 0.8085



Accuracy: 0.9635



Precision: 0.9251



Recall: 0.8708



F1: 0.8910



AUC: 0.9760

Visualizations

Evaluation images (evaluation_0.png, evaluation_1.png) show the input image, ground truth mask, and predicted mask for two test samples. These are generated during evaluation and saved in the output directory.

Requirements





Python: 3.11



PyTorch: 2.3.0 (with CUDA 11.8)



Dependencies:





torchvision==0.18.0



albumentations



scikit-learn



opencv-python



numpy



matplotlib



kagglehub

Install dependencies in a Kaggle notebook:

!pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
!pip install albumentations scikit-learn

Running the Code

On Kaggle





Create a Notebook:





Start a new Kaggle notebook.



Enable GPU (P100 recommended).



Enable internet access in notebook settings.



Add Dataset:





Attach the debeshjha1/kvasirseg dataset via the "Data" tab.



Install Dependencies:





Add a cell at the top of the notebook:

!pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
!pip install albumentations scikit-learn



Run the cell.



Run the Script:





Copy the code from kvasir_seg_pranet_max_metrics.py (available in this repository) into a notebook cell.



Run the cell to train and evaluate the model.



Outputs:





Training logs with per-epoch metrics.



Best model saved as best_pranet.pth.



Evaluation images (evaluation_0.png, evaluation_1.png) in the output directory.



Final test metrics printed.

Local Setup





Install dependencies using the above pip commands.



Download the Kvasir-SEG dataset manually or via kagglehub.



Update dataset_path in the script to point to the local dataset directory.



Run the script with Python: python kvasir_seg_pranet_max_metrics.py.

Files





kvasir_seg_pranet_max_metrics.py: Main script for training and evaluation.



best_pranet.pth: (Generated) Best model weights.



evaluation_0.png, evaluation_1.png: (Generated) Visualization of test predictions.

Future Improvements





Experiment with advanced architectures (e.g., DeepLabV3+, U-Net++).



Incorporate additional datasets for improved generalization.



Fine-tune hyperparameters (e.g., batch size, learning rate).



Add cross-validation for more robust evaluation.

Acknowledgments





Kvasir-SEG Dataset: Debesh Jha et al. for providing the dataset.



PraNet: Deng-Ping Fan et al. for the original model architecture.



Kaggle: For providing the computational environment.



PyTorch and Albumentations: For robust deep learning and augmentation libraries.

License

This project is licensed under the MIT License. See the LICENSE file for details.



Last updated: May 6, 2025
