# Master Thesis Data Science and Society 
Title: PREDICTING VASOVAGAL REACTIONS BASED ON FACIAL ACTION UNITS DURING BLOOD DONATION: A MACHINE LEARNING APPROACH
Student: Dionne Spaltman 
Summary: This project is part of the FAINT project, investigating the feasibility of predicting vasovagal reactions (VVRs) in blood donors using video recordings taken during the donation process. This research represents the first attempt to utilize video analysis for this purpose.

### Repository structure
This repository contains code for a machine learning pipeline with the following structure:
.
├── models
│   └── intensity_action_units          # The models that included both the VVR measurements from stage 1 and 2 and the action units data 
│       ├── nn.ipynb                    # Neural network 
│       ├── rf.ipynb                    # Random forest
│       ├── svm.ipynb                   # Support vector machine
│       └── xgboost.ipynb               # XGBoost
│   └── VVR_measurements                # The models with only VVR measurements from stage 1 and 2 as features 
│       ├── nn_12.ipynb                 # Neural network 
│       ├── rf_12.ipynb                 # Random forest
│       ├── svm_12.ipynb                # Support vector machine
│       └── xgboost_12.ipynb            # XGBoost
│   └── preparing_data.ipynb            # Creating the test and train sets and applying SMOTE
├── processing                 
│   └── action_units       
│       ├── cleaning_actionunits.ipynb 
│       ├── preprocessing_actionunits.ipynb
│       └── reducing_features.ipynb 
│   └── donor_info     
│       ├── cleaning_donorinfo.ipynb 
│       ├── preprocessing_donorinfo.ipynb
│       └── descriptives_donorinfo.ipynb
│   ├── descriptives.ipynb 
│   └── dimensionality_reduction.ipynb           
├── visualization                 
│   └── rfe.ipynb          
│   └── scatter_plot.ipynb
│   └── VVR.ipynb
└── README.md
