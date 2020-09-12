## Environment setup

* **git clone https://github.com/KutsynAndrey/2018_data_science_bowl_competition.git**  
* **cd 2018_data_science_bowl_competition**  
* **python3 -m venv venv**  
* **source venv/bin/activate**  
* **pip3 install -r requirements.txt**
* **extract files stage1_train.zip, stage1_test.zip and stage2_test_final.zip to project's root from competition's data page**
* **competition's data link: https://www.kaggle.com/c/data-science-bowl-2018/data**

## Solution
<p>The task was to create model for image segmentation on 2018 Data Science Bowl competition. During EDA it was noticed, that data has 3 fairly different classes that can be clustered using K-means algorithm. The clusters also have obvious imbalance in numbers of objects that can be a problem. One way to solve that can be returning balance to data using data augmentation techniques.<p>

<p> Also there was one image with incorrect mask, that should be erased from dataset before training<p>  
Image's id is "7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80".


## Files' descriptions

* **BasicDataExploration.ipynb** &mdash; jupyter notebook with observations on train data
* **data_preprocessing.py** &mdash; python script with data augmentation and loading.
* **generate_predictions.py** &mdash; python script for generating predictions based on pre-trained model.  
  Use it only if you have folder with pretrained model. Script takes image size as argument.  
  Use image size that was used for model training for correct predicting.  
  At line 30 you can change folder name for pre-trained model.  
  After generating in each image folder in stage1_train appears folder *predict* with Dice coefficient and predicted mask.
  
* **show_results.py** &mdash; script for displaying inputs and outputs of pre-trained model. Takes image size as argument.  
Use any button to skip image and "a" to close windows.
* **trainer_functional.py** &mdash; script with U-net's up and down layers. Also contains metrics, such as IOU and Dice coefficient.
* **trainer.py** &mdash; script for default training U-net model. Uses pre-defined augmentation that  
can be changed in *data_preprocessing.py* at line 108. Takes 4 arguments: *image size*, *number of epochs*, *batch_size*, *learning rate*.  
Order of arguments is important. In the end saves model to folder with name *my_model*. The name can be changed at line 42.
* **trainer_balanced_data.py** &mdash; trains model using balanced augmentation between 3 clusters.  
It means that augmentation step will produce more samples for clusters with less size. Takes the same arguments as *trainer.py*.  
Folder name &mdash; *my_balanced_model*
* **cluster_df.csv** &mdash; pre-clustered information for images in *stage1_train* folder.
* **filename.joblib** &mdash; pre-trained K-means model.
