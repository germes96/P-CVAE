# Variational autoencoder (VAE) for a Prototype-Based Explainable Neural Network

## Overview  🚀📚
This project implements a **prototype-based variational autoencoder (VAE**) architecture, designed to enhance both the performance and explainability of machine learning models. By integrating prototype-based methods with VAEs, the architecture provides intuitive and interpretable representations of data, making it easier to understand and analyze the learned features. The project includes multiple variants of the architecture, such as a classic VAE, a VAE with a classification layer, and a prototype-based VAE. It supports both CSV and Pickle formats for datasets.


## Models
The project includes the following architectures:
- **Classic Variational Autoencoder (VAE)**: This architecture is a standard VAE implementation, without any additional components.

- **Prototype-Based Variational Autoencoder (ProtoVAE)**: This architecture combines a VAE with a prototype-based method, where the prototypes are learned during training.

- **Variational Autoencoder with Classification Layer (CondVAE)**: This architecture adds a classification layer on top of the VAE, allowing for semi-supervised learning.

## Prerequisites
- Python 3.10.11
- PyTorch (version 1.12.1)
- NumPy (version 1.21.6)
- Pandas (version 2.0.2)
- Scikit-learn (version 1.2.2)
- Plotly (version 5.3.1)
- Tqdm (version 4.65.0)
- Kaleido (version 0.2.1)

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/germes96/P-CVAE.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd P-CVAE
    ```
3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Command-Line Arguments
- `-d` (str): Path to the dataset. Supported formats are `csv` and `pkl`. Default is `dataset/Gender/ALLIES/gender_x-vector_train.csv`.
- `-e` (str): Dataset format (`csv` or `pkl`). Default is `csv`.
- `-s` (int): Seed for the training unit where results will be stored. Default is `100`.
- `-n` (int): Number of training epochs. Default is `50`.
- `-t` (str): Specify the session type (`train` or `test`). Default is `train`.
- `-l` (int): Dimension of the latent space. Default is `2`.
- `-p` (int): Number of prototypes. Default is `-1` (set to the number of classes in the dataset).
- `--lr` (float): Learning rate for optimization. Default is `1e-3`.
- `--train-projection` (str): Whether to generate a projection for training data (`y` or `n`). Default is `n`.
- `--vae-type` (str): Type of variational encoder network (`v` for vanilla VAE, `c` for conditional VAE). Default is `v`.


### Datasets 📂
This project uses the following datasets:

1. **ALLIES Dataset**:
    - **The ALLIES dataset**.

2. **VoxCeleb Dataset**:
    - The VoxCeleb dataset is widely used for speaker identification tasks. It contains large-scale audio-visual data collected from interviews on YouTube.
    - **Reference**: 
      ```
      @inproceedings{Nagrani2017VoxCelebAL,
        title={VoxCeleb: A Large-Scale Speaker Identification Dataset},
        author={Arsha Nagrani and Joon Son Chung and Andrew Zisserman},
        booktitle={Interspeech},
        year={2017},
        url={https://api.semanticscholar.org/CorpusID:10475843}
        }
      ```
      For more details, visit [VoxCeleb official site](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

### Dataset Preparation 🛠️
1. **Create a `dataset` Folder**:
   - Before running the scripts, create a `dataset` folder in the root directory of the project.

2. **Unzip X-vectors**:
   - Download the x-vector files for the ALLIES and VoxCeleb datasets from the provided  [Google Drive link](https://drive.google.com/drive/folders/1fGfJHvAUdTDV4SZ8owOV550-4AjfibTk?usp=sharing).
   - Unzip these files into the `dataset` folder.

3. **Directory Structure**:
   - Ensure that the unzipped files follow the correct directory structure within the `dataset` folder. For example:
     ```
     project-root/
     ├── dataset/
     │   ├── VoxCeleb/
     │   └── Gender/
     │       └── ALLIES/
     └── ...
     ```

Make sure to set the correct paths to these datasets when running the scripts.



### Training
To train a model, run:
```bash
python main.py -d path/to/dataset.csv -e csv -n 100 -t train -l 10 -p 5 --vae-type v --lr 1e-3
```

### Testing
To test a model, run:
```bash
python main.py -d path/to/dataset.csv -e csv -t test -s 100 --vae-type v
```

### Example
#### Training with default settings:
```bash
python main.py
```

#### Testing with default settings:
```bash
python main.py -t test
```

## Model Checkpoints
Model checkpoints are automatically saved during training in the `models/` directory, under a folder named after the provided seed. The best model is saved as `checkpoint.pth.tar`.

## Utilities 🛠️📈
- **Training Logs** 📝: Saved during training, containing epoch information, losses, and accuracy.
- **Prototype Information** 🧬: Prototypes are saved as a CSV file in the model's directory for easy reference.
- **Classification Report** 📊: Generated after testing and saved in the model's directory, providing detailed metrics on model performance.


## Visualization 📊🎨
- **TSE Visualization**: Provides visualizations of the latent space using t-SNE, which are saved in the model's directory.
- **Latent Space**: The latent space of the model after testing is saved for further analysis and visual inspection.


## Warnings ⚠️
- **Dataset Format**: Ensure that the dataset is properly formatted and matches the specified format (`csv` or `pkl`).
- **Warnings Ignored**: The project ignores warnings by default for a cleaner output. Modify the code to enable warnings if needed.

## License
This project is licensed under the MIT License. 