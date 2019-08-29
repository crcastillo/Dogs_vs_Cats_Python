# Dogs_vs_Cats_Python
 
### Project Objective
The goal of this project is to build a Convolutional Neural Network with Tensorflow-GPU to classify images from
 [Dogs vs Cats Kaggle dataset](https://www.kaggle.com/c/dogs-vs-cats/data). This Sequential classifier consists of a 
 3 Convolutional layers and includes Pooling and Dropout layers. I perform some image pre-processing to allow for some 
 image rotation, zoom, and flipping, which should assist the classifier to understand typical image transforms.

### Required Packages
* datetime
* shelve
* os
* matplotlib.pyplot
* random
* numpy
* tensorflow.keras.models
* tensorflow.keras.layers
* tensorflow.keras.preprocessing.image
* tensorflow.keras.callbacks
* tensorflow.keras.models
* tensorflow.metrics
* sklearn.metrics

### Notes
* Getting tensorflow-gpu to run is a bit tricky, but ultimately worth it for the speed increase and I used the 
following setup:
  * PyCharm Community 2019.2
  * Conda 4.7.10
  * Python 3.7.3
  * Tensorflow 1.13.1
  * Tensorflow-GPU 1.14.0
* The speed increase from using the GPU reduced the training time to ~1/10 the CPU training time
* Training configuration
  * optimizer = adam (stochastic gradient descent)
  * loss = binary_crossentropy
  * metrics = accuracy
  * epochs = 50

### Interesting Findings
Despite trying to replicate the results of each run by establishing the random seed for both numpy and tensorflow, I 
can't perfectly replicate the validation results.

Validation Results Table (by Run)

| Training Run  | Accuracy  | AUC       | Avg. Precision-Recall  |
| :------------ | :-------: | :-------: | :--------------------: |
| Run 1         | 0.8433    | 0.9225    | 0.9092                 |
| Run 2         | 0.8102    | 0.9149    | 0.9059                 |
| Run 3         | 0.7800    | 0.9126    | 0.8993                 |
| Run 4         | 0.8250    | 0.9123    | 0.9013                 |
| Run 5         | 0.8176    | 0.8958    | 0.8885                 |

This appears to be the result of randomization that occurs within the initialization of weights by the GPU. This 
appears to be a [known problem](https://github.com/keras-team/keras/issues/2479) that can likely be addressed through utilizing a Theano backend and messing with the 
cuDNN parameters. I will look to address this in the next iteration.

The validated loss and accuracy metrics (*Run 5*) follow the training results fairly well as more epochs are added, but 
there is probably some additional room for tweaking the learning rate as training accuracy pulls away from validation accuracy. 
![Loss and Accuracy Plots](https://raw.githubusercontent.com/crcastillo/Dogs_vs_Cats_Python/master/Images/Loss%20and%20Accuracy%20Plots.png)

![Precision/Recall Curve](https://raw.githubusercontent.com/crcastillo/Dogs_vs_Cats_Python/master/Images/Precision-Recall%20Curve.png)

