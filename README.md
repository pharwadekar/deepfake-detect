# 50.021 Artificial Intelligence Project - Deepfake Detection

This repository is for a 50.021 Artificial Intelligence project at SUTD. We explore the ability of an deepfake detection model to generalise between deepfake video datasets using data augmentation and basic frequency analysis techniques. Following are brief instructions to get this code working.

## Environment and Setup

 1. Create a new virtual environment for this project. I use `pyenv` but use whatever suits you.

```bash
pyenv install 3.12.12
pyenv virutalenv 3.12.12 faker
pyenv activate faker
```

 2. Install PyTorch with reference to instructions [here](https://pytorch.org/get-started/locally/). You can skip to (3) but this will probably result in the PyTorch install being CPU only.
 3. Install the remaining dependencies using `pip install -r requirements.txt`

## Dataset

The datasets used to create this model are the FF++ dataset retrieved from [here](https://github.com/ondyari/FaceForensics) and the Celeb-DF dataset from [here](https://github.com/yuezunli/celeb-deepfakeforensics). FF++ was used for the training and validation sets while Celeb-DF was used to test the cross-dataset performance of the model. You can get the dataset from the original authors and extract the images used by the model for training and testing.

Or you can download the preprocessed data from [here](https://drive.google.com/drive/folders/1WC6YeGfzj9CFRydojA5oUV1A9fC0mJBV). The four zip files contain frames from the FF++ videos, faces extracted from those frames, frames from the Celeb-DF videos, and faces extracted from those frames.

 1. Create a new `data` directory using `mkdir data`
 2. Extract the datasets from the above drive link into `data`

## Training & Evaluation

Following the installation of dependencies and dataset set up, the notebooks in the `notebooks` directory can be explored to try training and testing the models. `Deepfake_Detection_Pipeline.ipynb` is the notebook designed for training and experimenting with models. Pretrained models can be loaded in `model_showcase.ipynb` to view their results. Instructions and details on the functions of each cell can be found within the notebooks.

## GUI playground

TODO