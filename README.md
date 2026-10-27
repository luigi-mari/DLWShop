# SciPy 2019 – Deep Learning Workshop: my notes

A neural network built from scratch (NumPy + automatic differentiation) to classify molecules as biodegradable or not.

This notebook, `06-keraslikemodel.ipynb`, was **coded along with the class** taught by [Eric Ma](https://github.com/ericmjl) at **SciPy 2019**. It is my own notebook, written while following the lesson, with short theory notes added later.

- Course materials: [ericmjl/dl-workshop](https://github.com/ericmjl/dl-workshop). The original notebook 6 is no longer in the repo, as the material has been reorganised since.
- SciPy 2019: <!-- add link to the tutorial page -->

## What's inside

- Random initialisation of a `[41, 20, 1]` network
- Dense layers with `tanh` and logistic activations
- Binary cross-entropy loss
- Gradient descent with automatic differentiation (`grad`)
- Confusion matrices on train and test data

## Why I'm sharing it

That workshop was my first contact with coding. I followed it out of curiosity, and little by little working with data became a real part of my work and my projects. Seven years later I'm adding the theory I didn't have then, and saying thank you.

## Running it

```bash
pip install numpy pandas scikit-learn autograd jax matplotlib seaborn tqdm
```

The data files (`biodeg_X.csv`, `biodeg_y.csv`) come from the workshop repo, in `data/`. Update the paths in the second cell to point to them.
