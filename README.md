RTAM
====

Ray Tracing for Accelerated Metrology

## Building

We use CMake to build the project. We are using `FindCUDA.cmake` and `FindOptiX.cmake`
from [Nvidia's OptiX samples](https://developer.nvidia.com/designworks/optix/download).
FindOptix doesn't always work as it assumes that this project is within OptiX's
SDK path. So we just need to set `OptiX_INSTALL_DIR` to the OptiX path. Once set
it should also set `OptiX_INCLUDE` correctly. If it's correct then you shouldn't
get any errors about finding `<optix.h>`.

If it can't find cuda, then try setting `CUDA_TOOLKIT_ROOT_DIR`, cuda is usually
located at `/usr/local/cuda` on Linux.

```
# change $HOME/optix to whereever you have optix installed.
cmake -B build -D OptiX_INSTALL_DIR=$HOME/optix
```
