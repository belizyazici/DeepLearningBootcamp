# Fish Species Classification 
This project aims to classify fish species using the "A Large Scale Fish Dataset" from Kaggle. The classification is carried out using an Artificial Neural Network (ANN) architecture. Below is a summary of the project steps.

## 1. Data Preparation
The dataset consists of folders containing fish images, where each folder represents a specific fish species.
Using Python's os library, the directories are traversed, and the file paths and labels of the images are collected into a Pandas DataFrame.
## 2. Data Preprocessing
The images are resized and normalized using ImageDataGenerator from TensorFlow's Keras library.
Data augmentation techniques like rotation, zoom, and horizontal flipping are applied to the training set to improve generalization.
## 3. Model Training
An Artificial Neural Network (ANN) model is built using Keras.
The model consists of multiple dense layers with ReLU activation functions and a final softmax layer for multi-class classification.
Dropout layers are added to prevent overfitting, and categorical cross-entropy is used as the loss function.
## 4. Model Evaluation
Accuracy and loss metrics are tracked during training.
Graphs are plotted for training and validation accuracy and loss to evaluate the model's performance.
A confusion matrix and classification report are generated to assess the model’s performance on the test set.
## 5. Hyperparameter Optimization
The model's performance is optimized by experimenting with hyperparameters like the number of layers, number of neurons, dropout rate, and optimizer choice.
## Conclusion
This project aims to build an accurate fish species classification model while preventing overfitting through proper preprocessing, model architecture, and hyperparameter tuning. The code and analysis provide a solid foundation for improving model performance. To take a deep look in the project you may check out my Kaggle notebook:<a href="https://www.kaggle.com/code/belizyazici/deeplearning-fishdataset/notebook" target="_blank" rel="noreferrer" style="color: #8e44ad;"> Deep Learning Fish Dataset </a>
