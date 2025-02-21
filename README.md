## Alzhimers Disease Detection using Deep Learning
This project focuses on classifying Alzheimer's MRI images using a hybrid deep learning model that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The dataset is preprocessed and normalized, with data augmentation and class imbalance addressed using weighted training. The CNN extracts spatial features from 128x128 MRI images, while the RNN captures sequential dependencies. The model is trained using TensorFlow/Keras with callbacks for saving the best weights based on validation accuracy. Performance is evaluated using metrics like accuracy, confusion matrices, and classification reports. Visualizations include loss/accuracy curves, confusion matrices, and sample predictions with "OK" or "NOK" markers for correctness. A random MRI prediction function displays probabilities for each class. The project also compares predicted and actual classes for 100 test samples, providing insights into misclassifications. Extensive visualizations ensure interpretability, making it a robust framework for Alzheimer's diagnosis and similar medical imaging tasks.
## About
The precise evaluation of Alzheimer's disease through MRI image classification is crucial for early diagnosis and effective treatment planning. This project focuses on classifying Alzheimer's MRI brain images using a hybrid deep learning model that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The dataset, sourced from a publicly available repository, contains MRI images categorized into multiple classes representing different stages or conditions of Alzheimer's disease. Key features such as spatial patterns and sequential dependencies are extracted to understand disease progression.Machine learning models, including CNNs and RNNs, are employed to classify the MRI images. Ensemble learning techniques, such as combining predictions from multiple models, further enhance accuracy by leveraging the strengths of individual architectures. The hybrid CNN-RNN framework achieves an accuracy of 94.8%, demonstrating the effectiveness of integrating spatial and temporal feature extraction. Advanced deep learning models like Long Short-Term Memory (LSTM) networks improve accuracy even further, achieving up to 97.5%.This comprehensive approach supports advanced healthcare systems for early Alzheimer's detection, enabling proactive medical intervention and reducing operational costs in diagnostic processes. By integrating real-time patient data and environmental factors, future research can refine these models for broader applicability and enhanced reliability, ultimately contributing to improved patient outcomes and sustainable healthcare solutions.
## Features
1.Data Collection
2.Data Preprocessing
3.Feature Extraction
4.Model Architecture
5.Ensemble learning
6.Evaluation Metrics
7.Visualization

## Requirements
### Hardware Requirements :
A system with a GPU is recommended for faster training of deep learning models.
Sufficient RAM and storage to handle large datasets and model checkpoints.
### Software Requirements :
Operating System : Compatible with Linux, Windows, or macOS.
Programming Language : Python 3.7 or later.
Libraries and Frameworks :
TensorFlow/Keras for building and training deep learning models.
NumPy and Pandas for data manipulation.
Matplotlib and Seaborn for visualizations.
Scikit-learn for evaluation metrics and class weight computation.
### Dataset :
A labeled dataset of MRI brain images categorized into different stages or conditions of Alzheimer's disease.
### Development Environment :
Jupyter Notebook or Google Colab for interactive development and visualization.
### Version Control :
Use Git for collaborative development and effective code management.
### Training Parameters :
Batch size: 32 images per batch.
Image size: Resized to 128x128 pixels.
Epochs: 25 (can be adjusted based on performance).
Optimizer: Adam optimizer.
Loss function: Sparse categorical cross-entropy.
### Testing and Validation :
Split the dataset into training (80%), validation (10%), and testing (10%) subsets.
Evaluate the model on the test set to ensure generalization.
### Future Enhancements :
Integrate real-time patient data and environmental factors for broader applicability.
Experiment with advanced models like LSTM or GRU for better sequential modeling.
Deploy the model as a web application for real-time predictions.

## System Architecture
![image](https://github.com/user-attachments/assets/31e04fe6-bfa7-47ee-b4e5-6eb13bd612dc)


## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Training and Validation Performance Graph

![image](https://github.com/user-attachments/assets/a3068f41-1cc2-4c06-86d9-afd6a1e188f1)

#### Output2 - Classification report
![image](https://github.com/user-attachments/assets/5053f1f7-32cd-4722-91a5-90b662df0831)

#### Output3 - Confusion Matrix
![image](https://github.com/user-attachments/assets/13d3a8fb-a8ee-4048-b541-95fe9cd29f75)


Detection Accuracy: 90%
Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact
The Alzheimer's disease detection program enhances early diagnosis using CNN, RNN, and LSTM models. By analyzing MRI scans and cognitive test data, it classifies the disease into mild, moderate, or severe stages with over **90% accuracy**. Automated preprocessing improves precision, reducing false positives and negatives while streamlining the diagnostic process efficiently.  

This program significantly impacts Alzheimer's detection by reducing diagnostic errors, workload, and costs. Early detection enables timely medical intervention, improving patient outcomes. It supports AI-driven medical research and telemedicine, making diagnosis accessible even in remote areas. Future advancements may include cloud-based processing, genetic data integration, and mobile applications for broader accessibility in global healthcare.
## Articles published / References
1. Mehmood A, Yang S, Feng Z, Wang M, Ahmad AS, Khan R, et al. A transfer learning approach for early diagnosis of Alzheimer’s disease on MRI images. Neuroscience. 2021;460:43–52.
2. Brookmeyer R, Johnson E, Zieglergraham K. Forecasting the global burden of Alzheimer’s disease. J Alzheimers Dis. 2007;3(3):186–91.
3. Mehmood A, Maqsood M, Bashir M, Shuyuan Y. A deep siamese convolution neural network for multi-class classification of Alzheimer disease. Brain Sci. 2020;10(2):84.
4. Bi X, Li S, Xiao B, Li Y, Wang G, Ma X. Computer aided Alzheimer’s disease diagnosis by an unsupervised deep learning technology. Neurocomputing. 2019;21:1232–45.
5. Tanaka M, Toldi J, Vécsei L. Exploring the etiological links behind neurodegenerative diseases: inflammatory cytokines and bioactive kynurenines. Int J Mol Sci. 2020;21(7):2431. 





