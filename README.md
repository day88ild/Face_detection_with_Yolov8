# Face Detection Project

This project is an experiment in face detection using YOLO. It includes a training scripts, and an inference script for real-time face detection. The project structure is as follows:

## Project Structure

- **/data**: This directory contains the dataset used for training and inference.
  - **/my_data_in_progress**: Placeholder for your data in progress.
  - **/the_main_data_yolo**: Main dataset for YOLO-based face detection.
    - **/full**: Contains the full dataset for face detection.
      - **/images**:
        - **/train**: Training images
        - **/val**: Validation images
      - **/labels**:
        - **/train**: Training labels
        - **/val**: Validation labels
      - **/configs.yaml**
    - **/part1** to **/part6**: Subsets of the main dataset, if applicable.
      - **/images**, **/labels** and **/configs.yaml**: Similar to the structure in `/full`.

- **/README.md**: This file, providing an overview of the project.

- **/getting_my_data.ipnb**: Jupyter Notebook for collecting and organizing data.

- **/model_training.ipnb**: Jupyter Notebook for training face detection model, and showing results of training.

- **/runs**: Directory for storing training progress and all supporting files including face detection models.
  - **/detect**
    - **/train** to **/train7**: Contains information about training process on different parts of data set (the data set was divided into 6 parts to speed up the training process) and **/train7** contains info about final training (1 epoch on the whole data set).
      - **/weights**: contains models params.
        - **/best.pt**
        - **/last.pt**
      - **/args.yaml**: contains additional information about training
      - and more (metrics, val_examples, train_batches).


- **/inference.py**: Python script for real-time face detection using a trained model.

- **/requirements.txt**: List of Python dependencies required for the project.

- **/blur_face.py**: similar to **/inference.py** but this one blur faces.


## Data

Unfortunately it is impossible to load the data I trained my models on due to Github limitations
but here you can find a link to it on Kaggle (https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset). 
And also feel free to create your own data using **/getting_my_data.ipnb** (your images will be saved into **/my_data_in_progress**) and you can then label it using label-studio

## Project Usage

1. **Data Collection**: Use `getting_my_data.ipnb` to collect and organize your data in `/data`.

2. **Model Training**: Train face detection models using `model_training.ipnb`.

3. **Inference**: Run `inference.py` to perform real-time face detection using a trained model.


## Dependencies

Ensure you have the required Python packages installed by running:

```bash
pip install -r requirements.txt


