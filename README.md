# Raycasting
Visualization of a set of spheres in 3D space using Phong reflection model.

## Requirements
* Nvidia GPU
* [Cuda Toolkit 12.6 or higher](https://developer.nvidia.com/cuda-toolkit)
* [CMake 3.25 or higher](https://cmake.org/)
* C++ 20 capable compiler (tested with MSVC)

## Build
In order to build the project execute the following commands:
```
git clone https://github.com/piotrprzybyszdev/Raycasting.git --recursive --shallow-submodules
cd Raycasting
cmake -S . -B .
```
Build files for the default build system of your platform should generate.
