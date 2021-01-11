# Project_recommender_system_movie_rating
## project description:
Recommender systems can be found everywhere in tody's world. be it movie recommendation on Netflix or products recommendations on Amazon,\
recommender system are making a significant impact, and it help users to easlity navigate through millions of products or tons of content(articles/videos/movies).
in this projects 


## Dependencies
Google colab
Tensowerflow 2.4

# Installation
```
 pip install -q tensorflow-gpu==2.0.0-rc0 
!pip install --upgrade tensorflow
!pip install tensorflow==1.2
!pip install tf-nightly
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
!apt-get -qq install -y graphviz && pip install pydot
import pydot
!apt-get -qq install python-cartopy python3-cartopy
import cartopy
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, \
  Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.models import Sequential

```

## Data info
Dataset source: http://files.grouplens.org/datasets/movielens
That data used in this project is a subset from open sourced movie lens datasetand contain 20million record for three column(User_id, movie_id, rating).\
we will train our recommender model using 80% of data and test it on the rest of the 20% user rating.

-Step1: laod the data:\
  we start jupyter notebook on colab and import the dependencies and libraties
  
 - Step2: Read the data set
 We then load and read teh dataset using pandas library.
 _ step3:Exploartory analysis
 
 - spiting the dataset:
   We have prepared the data for building the recommender model, we split the dataset inot trining and test sets. we split it into 80 to 20 ration\
   to train the model and test its accuracy.
   - Build and train the recommender model:
   We imort ----- from the ----- library and build model on the training dataset, there are multipule layers than can be tuned to imporve the performance of the model.
   -- Prediction and evaluation:
   
   ## issue
   None
   
   ### Contribution:
   you are welcome to pull request, and contribute to the project.
   
   
   ### Licence
   Copy right David Gabriel 
   
   ## Credit
   Google colab
   
   ### Reference:
   Intelligeent Pronect using Python.
   https://colab.research.google.com/


