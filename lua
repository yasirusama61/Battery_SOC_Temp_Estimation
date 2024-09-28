|-- ml-pipeline/
    |-- data/
        |-- raw/               # Raw data from AWS Timestream
        |-- cleaned/            # Cleaned and processed data
    |-- src/
        |-- data_collection.py  # Collect data from AWS Timestream
        |-- preprocessing.py    # Data cleaning and preprocessing steps
        |-- model_training.py   # Training your ML model
        |-- edge_deployment.py  # Code for deploying on the edge
    |-- models/
        |-- model.h5            # Trained model saved here
    |-- README.md               # Project overview and instructions
    |-- requirements.txt        # List of dependencies
    |-- .gitignore              # Files and directories to ignore
