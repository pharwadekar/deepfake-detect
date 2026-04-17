# 50.021 Artificial Intelligence Project - Deepfake Detection

## Environment and Setup

 1. Create a new virtual environment for this project. I use `pyenv` but use whatever suits you.

```bash
pyenv install 3.12.12
pyenv virutalenv 3.12.12 faker
pyenv activate faker
```

 2. Install PyTorch with reference to instructions [here](https://pytorch.org/get-started/locally/). You can skip to (3) but this will probably result in the PyTorch install being CPU only.
 3. Install the remaining dependencies using `pip install -r requirements.txt`