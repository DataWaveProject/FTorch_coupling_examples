# FTorch_coupling_examples

## Usage
These examples provide complete demonstrations of how to use FTorch to couple various neural network implementations in PyTorch using Fortran. 

- [CNN-simple](CNN-simple): Train a simple CNN to predict `y = x + 1` mapping where x is a 2D tensor of random values between -1 and 1 of size `(bs, 1, 20, 20)`, where domain size is 20 * 20. The example runs inference in Python and Fortran and checks that error is within an acceptable tolerance. 

- [FNO-1D](FNO-1D): Train a simple FNO-1D network to predict the sin wave `sin(2πx)` given an input of some dummy values and a uniform grid of x-positions on `[0, 1]`, and the target sine wave. The example runs inference in Python and Fortran with the same dummy values and unform grid. 

## Installation

Install FTorch as described [here](https://github.com/Cambridge-ICCS/FTorch), then compile the examples by following the README in each directory. For illustration, the cmake command will look something like the following (if using LibTorch - otherwise use PyTorch).

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/libtorch/share/cmake/Torch/;/path/to/FTorch/lib/cmake/FTorch/"
```

## Acknowledgements
This module includes an implementation of Fourier Neural Operators in `FNO-1d/fno-1d.py` adapted from **Pahlavan et al. (2024), On the importance of learning non-local dynamics for stable data-driven climate modeling: A 1D gravity wave-QBO testbed**, as made available in their accompanying repository [HamidPahlavan/Nonlocality](https://github.com/HamidPahlavan/Nonlocality/tree/main) Portions of the code are used under the MIT License. 
