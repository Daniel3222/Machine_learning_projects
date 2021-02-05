# Image classification from a sample of the Quickdraw dataset

This zipped folder which contains this README file is the result of the second Kaggle Data Competetion that just ended in IFT6390. It contains the following :

 - README
 - convnet_best.py
 - train.npz
 - test.npz
 - weights.best.hdf5 (weights used for the submission)
 
The python file is the script that will output the classification labels for the test data. It uses the train data (train.npz) as input for training and validation.

# Convolutionnal Neural Network : convnet_best.py


## Load data and split into train and val

We split the data, we then reshape it , we normalize the images and lastly we one-hot encode the labels vector into a matrix.

## Model building

#### Checkpoint
 - create a checkpoint for the best weights with a specific architecture of the model.
 - The checkpoint monitors the highest validation accuracy
 
#### Create model architecture
 - Create a sequential model
 - add necessary convolutions and other types of filters
 - use ImageDataGenerator to do data augmentation
 - Then do the fitting phase

## Prediction
 - rebuild the model architecture
 - load the weights
 - compile the model
 - load the test dat
 - normalize and reshape test data
 - predict classes of test data

## Creating CSV file

- Lastly, we put the data in a pandas dataframe that we will use as a data structure to create our csv file.

### Installation
Install the dependencies.

```sh
$ pip install numpy
$ pip install pandas
```

Now in order to create the submission.csv file that will contain the labels predicted, there are some important steps.
First, from your terminal, go into the folder containing all scripts and files. Then, one can simply write the following command

```sh
$ python path_to_submission_folder\submission_folder\convnet_best.py
```

The output of this line in the terminal will be the creation of a file called `<submission_ift_6390.csv>` in the folder where the code and the data is stored.

   
