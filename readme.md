## Transformer Parallelisms

This repository is an attempt to capture most common forms of transformer parallelization strategies using only [NumPy](https://numpy.org/) and [mpi4py](https://mpi4py.readthedocs.io/en/stable/tutorial.html). The goal is simple: create/add a parallelization strategy and ensure that a (tiny) model can train with it. We are not expecting any performance boost since the model is small and communication between devices will take time.

## Setup
Install the requirements 
```bash
pip install -r requirements.txt
```

## Model Parallel Strategies

### Tensor Parallel
Megatron-LM

### Sequence Parallel
Megatron-LM+

### Pipeline Parallel
GPipe, tera

## Zero Redundancy
