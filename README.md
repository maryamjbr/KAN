# Kolmogorov–Arnold Networks (KANs) Project

This repository contains an implementation of Kolmogorov–Arnold Networks (KANs) inspired by the paper titled "KAN: Kolmogorov–Arnold Networks". KANs provide a novel approach to neural networks by utilizing learnable activation functions on edges, offering improved accuracy and interpretability over traditional Multi-Layer Perceptrons (MLPs).

## Project Overview

The main objectives of this project are:
- **Implement KAN Architecture**: Develop the KAN model with B-spline activation functions.
- **Train and Evaluate KANs**: Train the KAN model on the Iris dataset and evaluate its performance.
- **Visualization**: Visualize the B-spline activation functions and interpret the results.
- **Unit Testing**: Ensure the correctness of the KAN components through unit tests.



### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/maryamjbr/KAN.git
   ```

2. Navigate to the project directory:

   ```bash
   cd KAN
   ```


### Running the Notebook

1. Start Jupyter Notebook:

   
   ```bash
   jupyter notebook
   ```

2. Open the notebook file:

   
   ```bash
   KAN.ipynb
   ```

3. Run the cells in the notebook sequentially to reproduce the analysis and results.


## Project Structure

```
├── data
│   └── iris.csv                 # Example dataset (if any)
├── models
│   ├── bspline_activation.py    # BSpline Activation function implementation
│   ├── kan_layer.py             # KAN Layer implementation
│   ├── kan_model.py             # KAN Model implementation
│   └── __init__.py
├── scripts
│   ├── main.py                  # Main script to run training and testing
│   ├── utils.py                 # Utility functions
│   ├── train.py                 # Training functions
│   ├── test.py                  # Testing functions
│   └── visualize_spline.py      # Script for visualizing the B-spline activation function
├── tests
    └──  test_kan.py              # Unit tests for KAN components

```


## Data

The dataset used in this project is the Iris dataset, which includes features such as sepal length, sepal width, petal length, and petal width, along with the target species.

## Results

### Model Performance

The KAN model demonstrates high accuracy in classifying the Iris dataset. Detailed results and performance metrics are printed during the training and testing phases.
```
Train Epoch: 1 [0/105 (0%)]	Loss: 2.547048

Test set: Average loss: 0.6332, Accuracy: 27/45 (60%)

Train Epoch: 2 [0/105 (0%)]	Loss: 0.651742

Test set: Average loss: 0.5606, Accuracy: 30/45 (67%)

Train Epoch: 3 [0/105 (0%)]	Loss: 0.500703

Test set: Average loss: 0.5295, Accuracy: 34/45 (76%)

Train Epoch: 4 [0/105 (0%)]	Loss: 0.302686

Test set: Average loss: 0.5087, Accuracy: 35/45 (78%)

Train Epoch: 5 [0/105 (0%)]	Loss: 0.416980

Test set: Average loss: 0.5082, Accuracy: 36/45 (80%)

Train Epoch: 6 [0/105 (0%)]	Loss: 0.416879

Test set: Average loss: 0.4738, Accuracy: 39/45 (87%)

Train Epoch: 7 [0/105 (0%)]	Loss: 0.571866

Test set: Average loss: 0.4739, Accuracy: 40/45 (89%)

Train Epoch: 8 [0/105 (0%)]	Loss: 0.491702

Test set: Average loss: 0.4505, Accuracy: 42/45 (93%)

Train Epoch: 9 [0/105 (0%)]	Loss: 0.564958

Test set: Average loss: 0.4331, Accuracy: 42/45 (93%)

Train Epoch: 10 [0/105 (0%)]	Loss: 0.471256

Test set: Average loss: 0.4237, Accuracy: 42/45 (93%)

Train Epoch: 11 [0/105 (0%)]	Loss: 0.357860

Test set: Average loss: 0.4080, Accuracy: 41/45 (91%)

Train Epoch: 12 [0/105 (0%)]	Loss: 0.340482

Test set: Average loss: 0.3749, Accuracy: 41/45 (91%)

Train Epoch: 13 [0/105 (0%)]	Loss: 0.200009

Test set: Average loss: 0.3585, Accuracy: 41/45 (91%)

Train Epoch: 14 [0/105 (0%)]	Loss: 0.568244

Test set: Average loss: 0.3575, Accuracy: 41/45 (91%)

Train Epoch: 15 [0/105 (0%)]	Loss: 0.152377

Test set: Average loss: 0.3388, Accuracy: 42/45 (93%)

Train Epoch: 16 [0/105 (0%)]	Loss: 0.128073

Test set: Average loss: 0.3225, Accuracy: 42/45 (93%)

Train Epoch: 17 [0/105 (0%)]	Loss: 0.326394

Test set: Average loss: 0.3004, Accuracy: 42/45 (93%)

Train Epoch: 18 [0/105 (0%)]	Loss: 0.125912

Test set: Average loss: 0.2804, Accuracy: 42/45 (93%)

Train Epoch: 19 [0/105 (0%)]	Loss: 0.207599

Test set: Average loss: 0.2738, Accuracy: 43/45 (96%)

Train Epoch: 20 [0/105 (0%)]	Loss: 0.203758

Test set: Average loss: 0.2767, Accuracy: 43/45 (96%)
```

### B-spline Activation Function Visualization

A visualization of the B-spline activation function used in the KAN model is provided for better interpretability.
![output](https://github.com/maryamjbr/KAN/assets/135154626/e53a4b75-5327-4ffc-b5f6-5ff00f37cca4)

