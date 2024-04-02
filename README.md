# Synthetic Data Generation using Variational Autoencoder (VAE) for Transformers with Nonparametric Variational Information Bottleneck

## Description

This algorithm focuses on utilizing a Transformer-based approach to generate synthetic data using Variational Autoencoders (VAEs) as the generative model. The VAE leverages the training dataset to create new samples (synthetic data). The integration of the Nonparametric Variational Information Bottleneck approach enhances the VAE's capabilities by regulating both the effective number of data vectors and the information conveyed by these vectors.

## Architecture and Algorithm

The proposed architecture combines Transformers and Variational Information Bottleneck (VIB) to address a key challenge: the variable-sized latent space introduced by Transformers. Unlike traditional VIB models, which operate on fixed-size vector spaces, this approach adapts to varying input sizes. 

The architecture comprises of mainly these sections:
1. **Nonparametric Bayesian Framework for Transformer Embeddings**: The main concept of this section is to represent the attention based information i.e focusing on the specific part of the input data as a mixture of vector space and consists of Nonparametric Bayesian Technique for modeling this information.
2. **Nonparametric Variational Information Bottleneck (NVIB)**: It is the extended version of the VIB that helps us to nonparameterized representation of the data by discarding the unnecessary details in the input vector. By the help of nonparametric prior and posterior values from the Transformer Embedings, NVIB is denoted as a regulariser that computes the KL divergence between the prior and posterior and the efficetiveness of the sample from the posterior of training.
3. **Nonparametric Variational AutoEncoder (NVAE)**:  NVAE helps to regulate the attention based representation between the encoder and the decoder in a Transformer.In this approach the transformer encoder is used to estimate the parameters (µq, µq, µq) of the posterior given the input whereas the transformer decoder is used to reconstruct the input text using denoising attention over the sample from this posterior. 
4. **Generative Modeling**: The generation of the new samples from the Transformer is by the usage of decoder from the trained transformer. Mainly it invloves these processes:
    - Sample a sentence length based on the distribution of sentence lengths in the training data
    - Sample from the prior conditioned on the chosen sentence length.
    - Use the trained Transformer decoder to generate a sentence based on the sampled prior.

## Mathematical Explanation
The mathematical interpretation of the the proposed algorithm can be illustrated as below:
- Denoising Attention: The attention function mainly provides a vector mapping a query vector to the  resulting attention vector. The attention vector provides the essential information of the 

## Evaluation Explanation
In order to support the theoritical baseline of this paper, experiments have been performed. 
- **Reconstruction Verses Generation**: VAE is the competetive in both reconstructing the input sentences and genrating new samples as the input sample.All the models involved in the experiment goes through the hyperparameter tuning
on the validation set across seeds inorder the find the best model that is capable of performing well in the given dataset.
- **Regularisation**: The NVIB layer is able to regularise the number of vectors in the latent representation of a NVAE. The NVAE models can automatically adjust the number of vectors by analyzing the text content, eliminating the need for manually programmed length functions like those used in VTS.

## Project Installation Guideline
1. Clone the repository from the github
```console 
git clone https://github.com/iamanupam1/VAE-For-Transformers-with-NVAE.git
```
2. Install conda and then mamba on your system using the bash script
```console
# Downloading the Miniconda from the Ananconda Repo and installing it
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local -f

# Installing Mamba using the Conda Environment
conda install -y mamba -c conda-forge
```
3. Change the directory and create a mamba environment
```console 
# Changing the Directory
cd nvib_transformers/
# Creating a Mamba Environment based on the environment.yml file in nvib_transformers
mamba env create -f environment.yml
```
4. Activating the Mamba Environment and listing the installed packges
```console
# Activating the Mamba Environment
mamba activate nvib
mamba list

```

# Your code implementation here
canvas = "Implementation"
