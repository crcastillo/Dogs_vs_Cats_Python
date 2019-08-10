#############################################################
#
#   Dogs vs Cats - Image Classification
#
#   Objective: Utilize Convolutional Neural Network to
#   train up an image classifier on Dogs vs Cats image
#   datasets. Familiarize myself with the techniques and
#   APIs that are commonly used to create these classifiers.
#
#   Initial Build: 9/16/2018
#
#   Notes:
#   - PyCharm Community 2019.2
#   - Developed w/ Conda environment 4.7.10
#   - Python 3.7.3
#   - Tensorflow 1.13.1
#   - Tensorflow-GPU 1.14.0
#############################################################



# Import required modules
# import warnings
import datetime as dt
import shelve # Save workspace objects library
import os
import matplotlib.pyplot as plt
import random
import numpy as np
# warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow import metrics
import sklearn.metrics
from inspect import signature
# warnings.resetwarnings()  # Reset warnings filter


# Establish the date of the script run
RunDate = dt.datetime.now()
# RunDate = dt.datetime(
#     2019
#     , 7
#     , 28
# )


# Ensure that AVX support is ignored to utilize the GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Set random.seed
Random_Seed = 123


#* Pick a random (sort of) sample dog picture and inspect
random.seed(Random_Seed)
sample = random.choice(os.listdir("Train_Data/dogs"))
image = load_img("Train_Data/dogs/" + sample)
plt.imshow(image)


# Define variables for image width, height, channels, and size
Image_Width = 128
Image_Height = 128
Image_Channels = 3
Image_Size = (Image_Width, Image_Height)


# Create Sequential object to define how our CNN will operate
classifier = Sequential()


# Add a convolutional layer to our neural network
classifier.add(
    Conv2D(
        filters=32  # Define number of filters
        , kernel_size=(3, 3)  # Define shape of each filter
        , input_shape=(
            Image_Width
            , Image_Height
            , Image_Channels
        ) # Define input shape for each image with resolution and define images as RGB (color)
        , activation='relu'  # Rectifier function
    )
)
classifier.add(  # Add
    BatchNormalization()  # Add a normalization layer, ensures mean activation close to 0 and std dev close to 1
)
classifier.add(
    MaxPooling2D(  # Add a pooling layer to get a more precise region where the features are located
        pool_size=(2, 2)
    )
)
classifier.add(
    Dropout(
        rate=0.25
    )
)


# Add second convolutional layer to our neural network
classifier.add(
    Conv2D(
        filters=64  # Define number of filters
        , kernel_size=(3, 3)  # Define shape of each filter
        , activation='relu'  # Rectifier function
    )
)
classifier.add(  # Add
    BatchNormalization()  # Add a normalization layer, ensures mean activation close to 0 and std dev close to 1
)
classifier.add(
    MaxPooling2D(  # Add a pooling layer to get a more precise region where the features are located
        pool_size=(2, 2)
    )
)
classifier.add(
    Dropout(
        rate=0.25
    )
)


# Add third convolutional layer to our neural network
classifier.add(
    Conv2D(
        filters=128  # Define number of filters
        , kernel_size=(3, 3)  # Define shape of each filter
        , activation='relu'  # Rectifier function
    )
)
classifier.add(  # Add
    BatchNormalization()  # Add a normalization layer, ensures mean activation close to 0 and std dev close to 1
)
classifier.add(
    MaxPooling2D(  # Add a pooling layer to get a more precise region where the features are located
        pool_size=(2, 2)
    )
)
classifier.add(
    Dropout(
        rate=0.25
    )
)

# Add a Flattening layer, which converts 2d image pixels to a singular vector
classifier.add(
    Flatten()
)
classifier.add(
    Dense(  # Add a fully connected (Dense) layer
        units=512 # Define number of nodes (input nodes > x > output nodes), common to use power of 2
        , activation='relu'  # Rectifier function
    )
)
classifier.add(  # Add
    BatchNormalization()  # Add a normalization layer, ensures mean activation close to 0 and std dev close to 1
)
classifier.add(
    Dropout(
        rate=0.50
    )
)
classifier.add(
    Dense(  # Initialize the output layer
        units=1,  # Binary classification = singular node
        activation='sigmoid'  # Sigmoid function
    )
)


# Compile the classifier
classifier.compile(
    optimizer='adam'  # Stochastic gradient descent
    , loss='binary_crossentropy'  # Loss function for the loss parameter, set for binary classification
    , metrics=['accuracy']  # Accuracy as the performance metric
)


# Define how Train images will be transformed/pre-processed ensure 80/20 Train/Validate split
Train_Datagen = ImageDataGenerator(
    rescale=1./255
    , rotation_range=15
    , shear_range=0.2
    , zoom_range=0.2
    , horizontal_flip=True
    , validation_split=0.2
)


# Define the location of the Training images
Training_Set = Train_Datagen.flow_from_directory(
    directory=os.path.abspath('Train_Data')
    , target_size=Image_Size
    , color_mode="rgb"
    , batch_size=32
    , shuffle=True
    , class_mode='binary'
    , seed=Random_Seed
    , subset='training'
)


# Define the Validation images to come from Training_Set
Validation_Set = Train_Datagen.flow_from_directory(
    directory=os.path.abspath('Train_Data')
    , target_size=Image_Size
    , color_mode="rgb"
    , batch_size=32
    , shuffle=True
    , class_mode='binary'
    , seed=Random_Seed
    , subset='validation'
)


# Define Train_Step_Size, Validation_Step_Size
Step_Size_Train = Training_Set.n//Training_Set.batch_size
Step_Size_Validation = Validation_Set.n//Validation_Set.batch_size


# Stops learning after 4 epochs
Early_Stop = EarlyStopping(
    monitor='val_loss'
    , patience=10
    , verbose=0
)


# Reduce learning rate by 0.5 if 4 epochs pass without accuracy of validation set improving
Learning_Rate_Reduction = ReduceLROnPlateau(
    monitor='val_acc'
    , patience=6
    , verbose=1
    , factor=0.5
    , min_lr=0.00001
)


# Train the model (classifier)
classifier.fit_generator(
    generator=Training_Set
    , steps_per_epoch=Step_Size_Train
    , validation_data=Validation_Set
    , validation_steps=Step_Size_Validation
    , epochs=50
    , callbacks=[Early_Stop, Learning_Rate_Reduction]
    , verbose=1
)


# Examine summary of model
classifier.summary()


# Save objects | NEEDS SOME WORK AS IT DIDN'T SAVE ALL OBJECTS
My_Workspace = shelve.open(
    filename=str(
        os.path.dirname(os.path.realpath('__file__'))
        + '\\'
        + RunDate.strftime("%Y%m%d")
        + '_Dogs vs Cats.out'
    )
    , flag='n'
)


# for key in dir():
#     try:
#         My_Workspace[key] = globals()[key]
#     except TypeError:
#         print('ERROR shelving: {0}'.format(key))
My_Workspace.close()


# Serialize Neural Network (CNN) model to JSON
classifier_json = classifier.to_json()
with open(
        str(
            RunDate.strftime("%Y%m%d")
            + "_classifier.json"
        )
        , "w"
) as json_file:
    json_file.write(classifier_json)
# Serialize weights to HDF5
classifier.save_weights(
    str(
        RunDate.strftime("%Y%m%d")
        + "_classifier.h5"
    )
)
print("Saved model to disk")


# Visualize the model training
fig, (ax1, ax2) = plt.subplots(
    nrows=2
    , ncols=1
    , figsize=(12, 12)
)
ax1.plot(
    classifier.history.history['loss']
    , color='b'
    , label='Training Loss'
)
ax1.plot(
    classifier.history.history['val_loss']
    , color='r'
    , label='Validation Loss'
)
ax1.set_xticks(
    np.arange(
        1
        , len(classifier.history.history['loss'])
        , 1
    )
)
# ax1.set_yticks(
#     np.arange(
#         0
#         , 1
#         , 0.1
#     )
# )

ax2.plot(
    classifier.history.history['acc']
    , color='b'
    , label='Training Accuracy'
)
ax2.plot(
    classifier.history.history['val_acc']
    , color='r'
    , label='Validation Accuracy'
)
ax2.set_xticks(
    np.arange(
        1
        , len(classifier.history.history['loss'])
        , 1
    )
)
legend = plt.legend(
    loc='best'
    , shadow=True
)
plt.tight_layout()
plt.show()


# Import model weights from JSON file and recreate model
classifier_json_file = open(
    file=#'20190728_classifier.json'
    str(
            RunDate.strftime("%Y%m%d")
            + "_classifier.json"
        )
    , mode="r"
)
classifier_model_json = classifier_json_file.read()
classifier_json_file.close()
classifier_model = model_from_json(classifier_model_json)


# Load model weights into classifier_model, should be identical to previously trained CNN model
classifier_model.load_weights(
    str(
        RunDate.strftime("%Y%m%d")
        + "_classifier.h5"
    )
)


# Define how the Test images will be transformed/pre-processed
Test_Datagen = ImageDataGenerator(
    rescale=1./255
)


# Define the location of the Testing images
Testing_Set = Test_Datagen.flow_from_directory(
    directory=os.path.abspath('Test_Data')
    , target_size=Image_Size
    , color_mode="rgb"
    , batch_size=32
    , shuffle=False
    , class_mode='binary'
)


# Create predictions on Testing_Set using loaded model from disk
Test_Predict = classifier_model.predict_generator(
    Testing_Set
    , verbose=1
)
Test_Predict = Test_Predict.reshape(Test_Predict.size, )


##############################################################
# Determine accuracy and other relevant metrics for Test Data
##############################################################

# Looks like this is meant to be utilized within train/test of potentially multiple parallel tensorflow models | Need to fix
# Test_accuracy, Test_update_op = metrics.accuracy(
#     labels=Testing_Set.labels
#     , predictions=np.around(Test_Predict) # Round the test predictions (currently a probability)
# )

# Accuracy = 0.8175976272862086
Test_accuracy = (Testing_Set.labels == np.around(Test_Predict)).sum() / Test_Predict.size
print(Test_accuracy)

# ROC_auc = 0.8994328200075845, good usage since there isn't much of a class imbalance
Test_auc = sklearn.metrics.roc_auc_score(
    y_true=Testing_Set.labels
    , y_score=Test_Predict
)
print(Test_auc)

# Determine average precision/recall, preferable when sizeable class imbalances exist
# average_precision_recall = 0.8941409661311999
Test_average_precision_recall = sklearn.metrics.average_precision_score(
    y_true=Testing_Set.labels
    , y_score=Test_Predict
)
print(Test_average_precision_recall)

# Build out precision/recall curve arrays
Test_precision, Test_recall, _ = sklearn.metrics.precision_recall_curve(
    y_true=Testing_Set.labels
    , probas_pred=Test_Predict
)

# Construct precision/recall curves
font = {
    'family' : 'DejaVu Sans'
    , 'weight' : 'normal'
    , 'size' : 20
}
plt.rc(
    group='font'
    , **font
)
plt.figure(
    figsize=(20, 10)
)
# step_kwargs = (
#     {'step' : 'post'}
#     if 'step' in signature(plt.fill_between).parameters
#     else {}
# )
plt.step(
    x=Test_recall
    , y=Test_precision
    , color='b'
    # , alpha=0.2
    , where='post'
)
plt.fill_between(
    x=Test_recall
    , y1=Test_precision
    , color='b'
    , alpha=0.4
    # , **step_kwargs
)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('2-Class Precision-Recall Curve AP = 0.894')