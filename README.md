# Dogs_vs_Cats_Python
 
### Project Objective
The goal of this project is to build a Convolutional Neural Network with Tensorflow-GPU to classify images from (Dogs vs Cats Kaggle dataset)[https://www.kaggle.com/c/dogs-vs-cats/data]. This Sequential classifier consists of a 3 Convolutional layers and includes Pooling and Dropout layers. I perform some image pre-processing to allow for some image rotation, zoom, and flipping, which should assist the classifier to understand typical image transforms.

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
* Getting tensorflow-gpu to run is a bit tricky, but ultimately worth it for the speed increase and I used the following setup:
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

### Interesting Findings
The initial validated loss and accuracy metrics follow the training results fairly well as more epochs are added, but there is probably some additional room for tweaking the learning rate as training accuracy pulls away from validation accuracy. 
![Loss and Accuracy Plots](https://raw.githubusercontent.com/crcastillo/Dogs_vs_Cats_Python/master/Images/Loss%20and%20Accuracy%20Plots.png)
