# Example 1 - FNO1d

This example provides a complete demonstration of how to use FTorch to run a trained Fourier Neural Operator network using Fortran. 


## Description

A Python file `fno1d.py` is provided that defines a typical FNO 'net' which is trained on a sampled sine wave. The task is to predict the 

It is then 

trained using `fno1d_train.py` that takes an input
vector of length 5 and applies a single `Linear` layer to multiply it by 2.

A modified version of the `pt2ts.py` tool saves this simple net to TorchScript.

A series of files `fno1d_infer_<LANG>` then bind from other languages to run the
TorchScript model in inference mode.

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

You can check that everything is working by running `fno1d_train.py`:
```
python3 fno1d_train.py
```
This instantiates the net defined in `fno1d.py` and trains, validates and outputs the network as either a `state_dict` or `TorchScript`. The `state_dict` option is compatible with `pt2ts.py`. 

The network expects inputs of the following form:
This simple case is not meant to challenge the FNO, but to validate the infrastructure:

- `input_tensor`: Dummy input, filled with zeros. Shape `(batch, x, 1)`.
- `grid_tensor`: x-positions on `[0, 1]`. Shape `(batch, x, 1)`.
- `target_tensor`: True values `sin(2πx)`. Shape `(batch, x, 1)`.

The FNO is thus trained to map a constant zero function (plus optional grid input) to a sine wave.

In many FNO implementations, especially those without fixed spatial encodings, the model input is formed by **concatenating the input function and positional grid** along the last (channel) dimension:
```
model_input = torch.cat([input_tensor, grid_tensor], dim=-1)
# giving shape (batch, x, 2)
```

If not already done so, save the `FNO1d` model to TorchScript by running the modified version of the
`pt2ts.py` tool:
```
python3 pt2ts.py
```
which will generate `saved_fno1d_model_cpu.pt` - the TorchScript instance of the net.

You can check that everything is working by running the `fno1d_infer_python.py` script:
```
python3 fno1d_infer_python.py
```
This reads the model in from the TorchScript file and runs it with an input tensor of shape `(1, x, 2)`.
The input_tensor (all zeros) and grid_tensor (`linspace(0,1,32)`) each have 32 elements.

It runs inference on the network and checks the error between the resulting outputs and known sine wave 
form are within an acceptable tolerance. 

At this point we no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved FNO1d model from Fortran we need to compile the `fno1d_infer_fortran` files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved FNO1d TorchScript from Fortran run the
executable with an argument of the saved model file:
```
./fno1d_infer_fortran ../saved_fno1d_model_cpu.pt
```

This runs the model with the array and repeats the example in `fno1d_infer_python` but in fortran. 

Alternatively we can use `make`, instead of CMake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.
```
make
./simplenet_infer_fortran saved_simplenet_model_cpu.pt
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
- Consider adapting the model definition in `fno1d.py` to function differently and
  then adapt the rest of the code to successfully couple your new model.
