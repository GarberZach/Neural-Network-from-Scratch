# Neural Network From Stratch

#### This project is my neural network created from scratch. It uses no machine learning libraries at any point. In this specific context it tackles a classifing problem. It creates a random 3 arm spiral made up by data points (imagine a bunch of data points on a 2-d graph, it looks like a spiral with 3 arms, kind of like the Milky Way Galaxy), all data points belong to one of these arms. The job of the neural network is to classify which of the arms each point belongs to. The logic for doing this is not my own. Additionally, the math for the back propagation, neuron weights, loss function, Rectified Linear Output (relo.py), and optimizers is also not mine (obviously haha), however, their implementation is. This network acheives an accuracy of 85% on test data, with managable over-fitting. 

## Note

This project will run if downloaded, provided appropriate packages have been installed. Navigate to Layer.py and run.


## Organization

  

  * Layer.py

     This files essentialy defines the network, here the amount of neurons for each layer is defined, and the methods for changing neuron weights and back propagation. Additionally, this files creates the data to train with and test data. In this file the optimizer, loss function, and Relo/softmax are called.

  * loss.py
    
    Provides a metric to gauge how "well" the network is doing in training, and methods for backwards and forward passing. To try to put it simply, these methods are for looking forward and backwards in time to change neuron weights.
     

  * Relo.py and softmax.py
  
     As training commences, neuron weights tend to exponentially increase, these classes keeps neuron weights small, keeping computation time down. However, only one of these functions may be used, not both.

  * Twitter_Stream.py and Tweet_Analyzer.py
  
     Preforms tasks very similar to WebScraper.py except using twitter as an article source.

  * AdaGrad_optimizer.py Adam_optimizer.py Adam_optimizer.py  and SGD_optimizer

    These are optimizers, there job is to govern the training process directly controlling which neuron weights are changed and when. Only one may be used at a time on any one training run. 

  * Results.txt

    This file contains a sample result from a training run incase running it on your computer proves to be too much work. 
