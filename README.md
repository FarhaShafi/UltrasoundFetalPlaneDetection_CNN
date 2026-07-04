# UltrasoundFetalPlaneDetection_CNN

### Overview
While doing ultrasound scanning of the pregnant woman, detecting the plane to identify the correct growth and measurement of the fetus is important. By doing this project, the model will predict the fetal plane and using this further while scanning we can use to calculate the growth of the fetus.

### Objective
To develop an accurate and efficient model that detects the fetal plane from the ultrasound images.

### Data Description
- Source: https://zenodo.org/records/3904280
          FETAL_PLANES_DB_data.csv
          Images - 12400
- 6 Classes:
    Fetal Anatomical Planes: Abdomen, Brain (Further categorized into the 3 most common fetal brain planes: Trans-thalamic, Trans-cerebellum, Trans-ventricular), Femur, Thorax.
    Mother’s Cervix.

- General Category: Including any other less common image plane.
- Meta Information: Patient number, US machine, Operator.
- Training-test split used in the Nature Sci Rep paper.

This repository contains the jupyter notebook file and its flask implemented front-end to test the prediction application.
The model detects the fetal plane from the given ultrasound image.

Repository contains the following folder:
    - FETAL_PLANES_ZENODO
        This contains the input images used for the training, validation and testing.
        Also contains FETAL_PLANES_DB_data.csv used for the training.
    - Output
        This folder will bbe created while running the notebook file, if it does not exists.
        This contains the saved model, metrics, plots and predictions obtained while training.
    - US_FetalPlaneDetection_Flask
        Flask implemented front-end for testing the model.

<img width="822" height="867" alt="image" src="https://github.com/user-attachments/assets/125133c6-fde0-4035-8aef-b03a4ca5ce51" />
