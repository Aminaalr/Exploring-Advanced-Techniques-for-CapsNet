Capsule Networks with Enhanced Architectures
This project introduces two advanced methods to improve the baseline Capsule Network (CapsNet) performance for image classification tasks. The proposed methods incorporate pre-trained architectures and novel Fire modules to address the limitations of conventional neural networks (CNNs) and CapsNet, enhancing feature extraction and dynamic routing mechanisms.
Table of Contents
Introduction
Proposed Methods
Pre-trained CapsNet Models
Fire Modules with Custom Swish Activation
Datasets
Performance and Results
Installation
Usage
Training and Evaluation
Results
References
Introduction
Capsule Networks (CapsNet) have emerged as a promising alternative to Convolutional Neural Networks (CNNs) for various classification tasks. However, CapsNet has several drawbacks, including computational inefficiency and information loss during max-pooling operations. This project introduces two advanced methods to improve the performance of CapsNet:
•	Pre-trained CapsNet Models using architectures like Google Network (Inception V3), Visual Geometry Group Capsule Network (VGG-CapsNet), and RES-CapsNet.
•	Fire Modules with Custom Swish Activation for efficient feature extraction and enhanced dynamic routing mechanisms.
•	Proposed Methods
•	Pre-trained CapsNet Models
•	Pre-trained architectures such as Google Network (Inception V3), Visual Geometry Group (VGG-CapsNet), and RES-CapsNet are utilized to extract rich features from images. Enhancements include:
•	Improved Squash Function: Stabilizes the learning process.
•	Modified Dynamic Routing Mechanism: Enhances the network's ability to capture complex patterns and relationships within the data.
•	Fire Modules with Custom Swish Activation
•	A novel method called Fire-CapsNet is introduced, employing Fire modules for efficient feature extraction. The custom swish activation function addresses the limitations of CapsNet.
Datasets
The proposed models are tested on four different datasets:
•	BM
•	MNIST
•	Fashion-MNIST
•	CIFAR-10
These datasets cover a wide range of application domains and help demonstrate the effectiveness of the proposed models.
Performance and Results
The models showed significant enhancements in classification performance and robustness against adversarial attacks. The evaluation on the BM dataset yielded the following results:
VGG-CapsNet: 99.31% accuracy
RES-CapsNet: 99.38% accuracy
GN-CapsNet: 99.89% accuracy
Fire-CapsNet: 99.92% accuracy.

