## Transformer Parallelisms

This repository is an attempt to capture most common forms of transformer parallelization strategies using only [NumPy](https://numpy.org/) and [mpi4py](https://mpi4py.readthedocs.io/en/stable/tutorial.html). The goal is simple: create/add a parallelization strategy and ensure that a (tiny) model can train with it. We are not expecting any performance boost since the model is small and communication between devices will take time.

The backbone of the project is the `numpy_distributed` library that [contains primitives](https://github.com/lweitkamp/transformer_parallelisms/blob/main/numpy_distributed/ops.py) for gathering, scattering, and reducing numpy `ndarrays`. A [sequential counterpart](https://github.com/lweitkamp/transformer_parallelisms/tree/main/numpy_sequential) implements the individual components of the transformer which is used as a baseline and for unit testing the distributed components.


## Setup
Install the requirements:
```bash
pip install -r requirements.txt
```

Run the unit tests:
```bash
mpirun -n 2 python -m pytest numpy_distributed/tensor_parallel/*_test.py
```



## Model Parallel Strategies

### Tensor Parallel
Megatron-LM

### Sequence Parallel
Megatron-LM+

### Pipeline Parallel
GPipe, tera

## Zero Redundancy
