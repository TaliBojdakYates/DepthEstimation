# Depth Estimation with ResNet

## Project Overview
This project focuses on using neural networks to estimate depth with monocular cameras. The study evaluates various Residual Neural Network (ResNet) architectures to determine the most effective models for depth estimation.

## Objectives
The project's primary goals are:
- To evaluate the accuracy of depth estimation using different ResNet models.
- To experiment with loss functions to minimize overfitting.
- To explore methods to enhance real-time processing for autonomous systems.

## Components
- **Data Preparation**: Preparing the dataset for training,testing, and validation.
- **Neural Network Models**: Evaluating ResNet architectures (e.g., ResNet-50V2, ResNet-101V2) based on metrics like absolute relative error (ARE) and root mean squared error (RMSE).
- **Loss Functions**: Using different loss functions, including Huber loss, gradient loss, and structural similarity index measure (SSIM), with specific weights to prevent overfitting.

## Key Findings
- The ResNet-50V2 model offers a good balance between accuracy and training time.
- The ResNet-101V2 model has higher accuracy but exhibits more overfitting issues.


## Recommendations
- **Improving Data Quality**: Focus on enhancing dataset quality by addressing errors and missing values.
- **Enhancing Real-time Processing**: Simplify model architecture to improve real-time processing capabilities for autonomous systems.
- **Exploring Alternative Regulation Methods**: Investigate other methods to increase robustness beyond elastic net regulation.

## Conclusion
The project demonstrates the potential of neural networks with monocular cameras for depth estimation. It shows the importance of choosing the right model architecture and loss functions to achieve accurate results and avoid overfitting. Future work could focus on improving data quality and exploring more efficient model architectures.
