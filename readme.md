## NuMPItron
Implementation of Megatron and various other parallelization strategies for transformer models using only [NumPy](https://numpy.org/) and [mpi4py](https://mpi4py.readthedocs.io/en/stable/tutorial.html).

The backbone of the project is the `distributed` library that [contains primitives](https://github.com/lweitkamp/transformer_parallelisms/tree/main/distributed/ops.py) for gathering, scattering, and reducing numpy `ndarrays`. A [sequential counterpart](https://github.com/lweitkamp/transformer_parallelisms/tree/main/layers) implements the individual components of the transformer which is used as a baseline and for unit testing the distributed components.


## Setup
Install the requirements:
```bash
pip install -r requirements.txt
```

Run the unit tests:
```bash
mpirun -n 2 python -m pytest numpy_distributed/tensor_parallel/*_test.py
```

## Checklist

- [ ] Sequential
- [ ] Tensor Parallel
- [ ] Sequence Parallel
- [ ] Data parallel
- [ ] Tera Pipeline Parallel
- [ ] ZeRO?
