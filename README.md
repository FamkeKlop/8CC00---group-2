# 8CC00---group-2
## Advanced Programming

### Introduction and scope
This project focussed on the implementation of a **machine learning model** with respect to predicting **kinase inhibition**. The model could be used as a means for future **drug discovery**, which up until now has been a time-consuming process. Computer aided drug discovery (CADD), will allow researchers to find more molecular candidates for medical purposes in a more timely manner. To this end multiple appraoches have been implemented. This page contains all of those, and some unfinished or unused codes that were not deemed significant enough to mention in the final product. However, some of this code could be used in further research as they might aid further accurate prediction models.

### To get started
The complete code is divided into 7 main sub categories each contained in a folder
#### 1. Generate_input
This folder contains the dataset with which the model is trained, and the dataset on which the final results are tested (what is handed in in the final assignment). Using this dataset all features are generated using the RdKit. Other files in this folder include a piece of code that  generates a dataset with only a couple of all of the features. This dataset was occasionally used for quickly testing machine learning inputs.

#### 2. EDA
This folder contains all the code and files that are associated with the exploratory data analysis. These include: 
- A code that filters for outliers using an isolation forest, called *Cleaning_dataset.py*
- A code that filters out correlated varaibles, called *Correlations.py*
- A code that filters out variables with a low variance, called *low_variance_filter.py*
- A code that generates the boxplots found the in "/Boxplots" folder, called *boxplots.py*
- A couple of HTML files that contain the descriptive statistics for different inhibition groups based on an analysis done using the Jamovi software.

#### 3. Fingerprints
This folder contains two main files: A code that generates Extended Connectivity Fingerprints (ECFP) for each molecule and the accompanying dataset that is created.

#### 4. Boxplots
This folder contains the images of the boxplots that are generated using the boxplot code. It visulizes the distributions of all features.

#### 5. ML
The machine learning folder contains

#### 6. Prediction
#### 7. Visulalizations


### Final Remarks
All contributors to this project are: Laura Moiana, Giorgia Barra, David Keur, Kirsten Brusse, Jaleel Dibani, Roan Jacobs, Jort Lokers and Famke Klop.




