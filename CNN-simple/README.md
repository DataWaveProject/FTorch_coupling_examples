# Example 1 - CNN-simple

This example provides a complete demonstration of how to use FTorch to run a trained Convolutional Neural Network (CNN) using Fortran. 


## Description

A Python file `cnn-simple.py` is provided that defines a typical CNN trained to compute the mapping y = x + 1. 

## Dependencies

This example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation.
Then from this directory create a virtual environment and install the necessary Python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check that everything is working by running `cnn-simple.py`:
```
python3 cnn-simple.py
```
This instantiates the CNN which consists of a couple of 2D convolutions, trains it and then outputs the network in the `TorchScript` format. The kernel essentially learns to act like an identity filter (1 in middle, 0 elsewhere in 3 x 3 ) and the bias will converge to 1. 

The network expects inputs of the following form:

- `input_tensor`: Initialised as random values between -1 and 1. Shape `(batch_size, 1, 20, 20)`.
- `target_tensor`:  (input_tensor + 1). Shape `(batch_size, x, 1)`.

You can check that everything is working by running the `cnn-simple_infer_python.py` script:
```
python3 cnn-simple_infer_python.py
```
This reads the model in from the TorchScript file and runs it with an input tensor of shape `(batch_size, 1, 20, 20)`.

It runs inference on the network and checks the error between the resulting outputs and incremented ground truth are within an acceptable tolerance. 

At this point we no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved CNN-simple model from Fortran we need to compile the `CNN-simple_infer_fortran` files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved CNN-simple TorchScript from Fortran run the
executable with an argument of the saved model file:
```
./cnn-simple_infer_fortran ../saved_cnn-simple_model_cpu.pt
```

This runs the model with the array and repeats the example in `cnn-simple_infer_python` but in fortran. 

Alternatively we can use `make`, instead of CMake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.
```
make
./cnn-simple_infer_fortran saved_cnn-simple_model_cpu.pt
```

You will also likely need to add the location of the dynamic library files
(`.so` or `.dylib` files) that we will link against at runtime to your `LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
```
or `DYLD_LIBRARY_PATH` on mac:  
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:</path/to/library/installation>/lib
```

## Further options

To explore the functionalities of this model:

- Try saving the model through tracing rather than scripting by modifying `pt2ts.py`
- Consider adapting the model definition in `cnn-simple.py` to function differently and
  then adapt the rest of the code to successfully couple your new model.
