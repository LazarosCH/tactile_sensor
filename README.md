# Install the package

1. Install the required packages, in the versions listed in the [requirements.txt](requirements.txt).
2. Install the cuda-toolkit:
```sh
sudo apt install nvidia-cuda-toolkit
```
check installation with:
```sh
which nvcc
ptxas --version
```

- if you have gpu installed, these packages (NVCC (Nvidia CUDA Compiler) and assembler for Parallel Thread eXecution) are required for faster (x8) inference