
---

# Duplicate Bug Report Detection (DBRD) using Siamese Neural Networks with Spatio-Temporal Locality

This repository contains the code and data used in our research on improving Duplicate Bug Report Detection (DBRD) using Siamese Neural Networks with Spatio-Temporal Locality.

## Data

We utilized the dataset provided by Lazar et al., which can be found [here](https://alazar.people.ysu.edu/msr14data/#). It includes bug reports from three open-source projects: OpenOffice, Eclipse, and NetBeans. The processed versions of the datasets for training are stored in `.rar` format. Simply download and extract them to use.

- `deal_datasetexample.ipynb`: An example notebook showcasing the process of handling the datasets.
- `component.ipynb`: A notebook demonstrating the analysis of certain attributes.

## Models

- `DBRD.ipynb`: This notebook presents the network model structure used in our research.
- `embedding.ipynb` and `tokenize.ipynb`: These notebooks show how to obtain corresponding embedding vectors through BERT pre-processing.
- `normalization.ipynb`: A notebook illustrating the process of handling penalty terms.

## Baseline Models

- `bert-mlp.py` and `dc-cnn.py`: These files contain the model structures for the baseline models.

## Data Preparation and Testing

- `dataprepara-dccnn.py`: A script showing how to prepare the dataset for input into the DC-CNN model.
- `test.ipynb`: A notebook for testing the baseline models with the prepared dataset.

---

