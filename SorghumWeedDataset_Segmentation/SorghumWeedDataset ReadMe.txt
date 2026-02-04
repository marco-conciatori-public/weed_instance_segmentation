===============================
SorghumWeedDataset_Segmentation
===============================

Description:

Purpose of dataset creation:
‘SorghumWeedDataset_Segmentation’ is created to address real-time weed challenges precisely and encourage weed research using computer vision applications.

About the dataset:
‘SorghumWeedDataset_Segmentation’ is a crop-weed research dataset with 5555 manually pixel-wise annotated data segments from 252 data samples which can be used for object detection, instance segmentation, and semantic segmentation. The data segments consist of sorghum samplings (Class 0), Grasses (Class 1), and Broad-leaf weeds (Class 2) which are the three research objects focused during this data acquisition process. The TVT (Train: Validate: Test) ratio is set as 8:1:1 to split the data samples into training, validation, and testing. The ground truth preparation is carried out by manually annotating the data segments pixel-wise, using VIA (VGG Image Annotator) software. The respective annotation files for training, validation, and testing are provided in JSON, CSV, and COCO formats. 

Equipment used for data acquisition: 
To record a rich set of information on the research objects, a state-of-the-art instrument - Canon EOS 80D – a Digital Single Lens Reflex (DSLR) camera with a sensor type of 22.3mm x 14.9 mm CMOS is used.

Data type, format, and size: 
Each data sample is an RGB image represented in JPEG format with 6000 × 4000 pixels making an average size of 13MB. This rich set of information from the data sample assisted in annotating data segments of plant length lesser than 0.5cm.

Data acquisition: 
This dataset emphasizes the early stages of crop growth to meet the challenges faced during the ‘Critical period of weed competition’. Data samples are captured from agriculture fields that follow both uniform crop spacing and random crop spacing. 

Temporal coverage: 
Data is acquired during April and May 2023. To generalize the dataset, data is acquired in various light and weather conditions with varying distances.

Geographical coverage: 
Data is acquired from Sri Ramaswamy Memorial (SRM) Care Farm, Chengalpattu district, Tamil Nadu, India. To the best of our knowledge, ‘SorghumWeedDataset_Segmentation’ is the first open-access crop-weed research dataset from Indian fields for segmentation that deals with weed issues in uniform and random crop-spacing fields.

Expected outcome:  
The expected outcome of this dataset will be an Artificial Intelligence (AI) model that localizes and segments all the research objects present in a particular data sample.

Detailed description: 
A detailed description of the dataset and data acquisition process is given in the data article entitled “ ‘SorghumWeedDataset_Classification’ And ‘SorghumWeedDataset_Segmentation’ Datasets For Classification, Detection, and Segmentation In Deep Learning “. (Submitted in the journal ‘Data in Brief’ on 25/09/2023 and awaiting publication)

Citation: 
If you find this dataset helpful and use it in your work, kindly cite this dataset using “Michael, Justina; M, Thenmozhi (2023), “SorghumWeedDataset_Segmentation”, Mendeley Data, V1, doi: 10.17632/y9bmtf4xmr.1”

Further queries: 
If any queries/suggestions concerning this dataset, please e-mail us at thenmozm@srmist.edu.in [corresponding author] 
