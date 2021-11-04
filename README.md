<h1>Create and train with tfRecords</h1>

The code can be used to save preprocessed or unpreprocessed images with their corresponding label as tfRecords file. 

<h2>How to use the code</h2>

1. Split your dataset into train, test and validation subset
2. Create 2 configuration dictionaries within the configuration.py file. The train config keeps track of all the necessary information that is bound to a specific architecture and the data set config the information for a specific dataset. I've implemented the imagenet data set, which was stored on the IKW compute cluster and several ann architectures, which can be loaded from keras.applications. Further information of how to structure the dictionaries can be found within the respective file.
3. Create tfRecords for each data set subset by running the tfRecords.py file with the corresponding command line attributes
    e.g. for the val dataset, we want the data to be saved preprocessed and have 50 files per shard: python tfRecords.py -dt val -s 50 -p True
4. Look at the train_tfRecords.py file. It gives an examplary usecase for training a artificial neural network with the saved tfRecords files.
