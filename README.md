# Quantum RNNs

This research track is about our investigation on Quantum Recurrent Neural Networks (QRNN), improving on well-known classical machine learning techniques like Long Short Term Memory (LSTM) or Gated Recurrent Units (GRU) using the power of Quantum Neural Networks (QNN) to create quantum counterparts QLSTM and QGRU.

We have done a couple of use cases for it so far:

QLSTM for asset price prediction: https://github.com/DikshantDulal/SoftServe_QLSTM
QGRU for quantum drug generator: https://gitlab.com/damyr_hadiiev/quantum-drug-generator/-/tree/main/ReLeaSE-multiobjective
We have found a couple of academic papers that would be useful:

The power of quantum neural networks by Abbas et al. https://arxiv.org/pdf/2011.00027.pdf
Effective dimension of machine learning models by Abbas et al. https://arxiv.org/pdf/2112.04807.pdf

We are consolidating the effort into the following GitHub: https://github.com/JonasTYW/Quantum-RNNs

# Workflow and To-do

Here we discuss the plan for the paper for QRNNs. Below shows the general workflow for QRNNs:

![Workflow](img/workflow.JPG)

We split it up into three sections: the preparation of data; the implementation of the QRNN (or RNN) itself; and the metrics for the model.

## Data preparation

Being a machine learning technique, data preparation is still extremely important for QRNNs despite being classical. In fact, to a certain extent, due the limitations of QML models (and by extension, QRNN models), careful preparation of data becomes a whole lot more important. This directly affects the quality of the model that would be created, so we need to approach this carefully.

### Classical dataset loading

The first question one might ask is why we are only considering classical datasets. The simple answer to the question is that for the foreseeable future, most of the use cases that we would be interested in comes in the form of classical data. Beyond that, quantum datasets are also notoriously difficult to understand from an epistemological point-of-view, and the simplest (and by simplest I mean most obvious) use of quantum data would likely only come after a QRAM device has been made, that is to say beyond NISQ-era. Thus, we focus our current efforts on classical data.

This stage of data preparation is more of considering the architecture of the code, and how we want to load the dataset to be trained in the quantum model. There are a few preset functions that already do this: the Dataloader in the Pytorch library, Keras also has it's dataloader function. However, during the course of the two use cases we explored above, we found out that we had to do extra preparation for loading. For asset price prediction, we did some extra preparation to put it into the Dataloader class from Pytorch, whereas for the quantum drug generator we instead used the dataloader function from the ReLeaSE algorithm. This section is thus architectural in nature, where we consider how we want to load the data, and whether there's a good way to unify the architecture to make it easier for QRNNs in general.

### Exploratory Data Analysis

This section is fairly self-explanatory, and also not much different from classical machine learning. Basic things we need to check are correlation between variables, checking for outliers, and removing any suspicious looking data. Visualizations of data provided would also be good. The main point of this is to clean the data to make sure that it is ripe for machine learning. A good model cannot save bad data. Beyond that, the EDA would also help to inform our data preprocessing methods below.

### Data Preprocessing methods

Methods that we are thinking about:

- Normalization: Make sure that variables are normalized so that all variables are of similar scale.
- PCA: Linear Dimension reduction technique that maximizes variance. Applicable to most datasets.
- LDA: Linear Dimension reduction technique that maximizes difference between classes. Applicable to only classification problems.
- NLDR: Non-linear Dimension reduction techniques. If we need to go there, I will look into it more, but methods like kernal PCA or different type of mappings can be used.

## QRNN Implementation

### Quantum Encoding

There are two considerations that we have here:

Type of Encoding:
- Amplitude Encoding
- Phase Encoding
- Other encoding schemes

Within the type of encoding, we also have to decide:

- Number of rotations
- Entangling?

### Type of Backend

### VQC structure 

Structure of VQC

Basically, how do we want the circuit to look? Fully entangling, how many rotations, how to entangle, how many qlayers?

### Type of Model and Optimizer

Structure of Neural Network Model

Type of RNN:
- GRU
- LSTM
- QGRU
- QLSTM

Preprocessing Layer:
- How many layers
- Structure of layer

Postprocessing Layer:
- How many layers
- Structure of layer

Optimizers:
- Adam
- Adadelta
- etc

### Hyper-parameter Search

Likely have to perform a grid-search for LR

## Metrics

### Loss

### Visualizations 

### Number of parameters used

### Time analysis

### Information-theoretic bounds

## Testing for the paper

### Experimental

### Theoretical