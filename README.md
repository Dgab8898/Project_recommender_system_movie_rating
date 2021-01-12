# Project_recommender_system_movie_rating
## project description:
Recommender systems can be found everywhere in tody's world. be it movie recommendation on Netflix or products recommendations on Amazon,
recommender system are making a significant impact, and it help users to easlity navigate through millions of products or tons of content(articles/videos/movies).\

In this projects we build a recommender system model using Deep learning and neural networks.


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
That data used in this project is a subset from open sourced movie lens datasetand contain 20million of movie rating. the record consists of three column(User_id, movie_id, rating).\
we will train our recommender model using 80% of data and test it on the rest of the 20% user rating.

-Step1: import the libraries:\
  we start jupyter notebook on google colab and import all dependencies and libraties.
  
 - Step2: Read the data set
 We then load and read teh dataset using pandas library.
 _ step3:Exploartory analysis
  wee explore our data
  
 _ feature engineering:
 We cast user id column into categorical, doing this authomatically assigned integer in coding, starting from zero to each user id. then we assinged this code to new column called new user id. we do the same to Movie id.\
 then we convert user_id and movie id to ``nparray``. then get the number user and movie id
 

   - Build the  model and compile it
   - Make our neural network:
   - we use the user and movie as an inputs, then we create two embeddings, and fkatten both embedding
   -  The concatenat the user-movie int feature vector 
   
    - spiting the dataset:
   We have prepared the data for building the recommender model, we split the dataset inot trining and test sets. we split it into 80 to 20 ration\
   to train the model and test its accuracy.
   We imort keras from the tensoflowlibrary and build model on the training dataset,\
   there are multipule layers than can be tuned to imporve the performance of the model.
   -- fi the model we passing training and we set as x target for user and mvoie training an y target for rating treaing, we set the epoch for 25, and\
   the size of 1024, veebose= 2 so we can train faster in google colab.We set validation on testing data,
   - By the end we get 0.6259 of error
   
   ## issue
   None
   
   ### Contribution:
   you are welcome to pull request, and contribute to the project.
   
   
   ### Licence
   Copy right David Gabriel 
   
  
   
   ### Reference:
   Intelligeent Pronect using Python.
   https://colab.research.google.com/
   https://www.udemy.com/course/deep-learning-tensorflow-2/learn/lecture/16271032#overview


