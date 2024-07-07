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

   ```

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
│   └──  test_kan.py              # Unit tests for KAN components

```


## Data

The dataset used in this project is the Iris dataset, which includes features such as sepal length, sepal width, petal length, and petal width, along with the target species.

## Results

### Model Performance

The KAN model demonstrates high accuracy in classifying the Iris dataset. Detailed results and performance metrics are printed during the training and testing phases.

### B-spline Activation Function Visualization

A visualization of the B-spline activation function used in the KAN model is provided for better interpretability.
