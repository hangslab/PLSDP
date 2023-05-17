# PLSDP
Protein subcellular localization based on deep learning.
# Requirements

- MULocDeep dataset: [GitHub - yuexujiang/MULocDeep](https://github.com/yuexujiang/MULocDeep)

- CCNet: [GitHub - speedinghzl/CCNet: CCNet: Criss-Cross Attention for Semantic Segmentation (TPAMI 2020 & ICCV 2019).](https://github.com/speedinghzl/CCNet)

- Python 3.7.4.

- Keras version: 2.3.0

- For predicting, GPU is not required. For training a new model, the Tensorflow-gpu version we tested is: 1.13.1

- Users need to install the NCBI Blast+ for the PSSM. The download link is [https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/).

  

# Running on GPU or CPU

If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions.

# Introduction

The dataset is quantitatively characterized using coding methods such as BLOSUM62 scoring matrix, position-specific scoring matrix, amino acid physical and chemical properties and word embedding. Next, the data are characterized by feature extraction using the idea of jump connection in convolutional neural networks and residual networks, and are effectively combined with GRU network models. In order to maximize the use of long sequence feature information as much as possible, we apply the multi-headed self-attention mechanism and the criss cross-attention mechanism to collect the information of the elements in the ranks and columns where the elements are located through cyclic operations to achieve the capture of contextual information. 

We finally built the following models：
 - CNN+BiGRU+PSSM model
-  CNN+BiGRU+CBOW model
-  BiGRU+Multi-head self-attention+PSSM model
-  BiGRU+Multi-head self-attention+CBOW model
-  ResNet+BiGRU+CBOW model
 - BiGRU+Criss Cross-Attention+PSSM model

# Acknowledgement

We  appreciate the help from [GitHub - yuexujiang/MULocDeep](https://github.com/yuexujiang/MULocDeep) and [GitHub - speedinghzl/CCNet: CCNet: Criss-Cross Attention for Semantic Segmentation (TPAMI 2020 & ICCV 2019).](https://github.com/speedinghzl/CCNet)

# Contacts

If you have any questions or problems using our tools, please contact us. (Email: hangs@xtu.edu.cn)
