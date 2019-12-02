Plasticity
==========

Plasticity is a neural network and symbolic math framework written in C++. The
framework aims to give developers a simple interface to design and train neural
networks on specialized hardware.

Plasticity runs on OpenCL and can support various OpenCL backends.
(ARM, Intel CPUs, NVIDIA & AMD GPUs, POCL, etc). Plasticity is being developed
on both NVIDIA's OpenCL backend and POCL, and these are the two backends that
are going to receive the most software support.

Directory Structure
-------------------

```
codegen/ - Utility code to simplify generating the OpenCL kernels.
filter/ - A kalman filter implementation based on the symbolic library.
geometry/ - Matrix, vector, and quaternion math routines.
memory/ - A wrapper for OpenCL buffers.
nnet/ - The neural network library.
stats/ - Statistics-related coroutines and helper functions.
symbolic/ - A symbolic math library with automatic differentiation.
third_party/ - External 3rd-party code imported for this project.
```

Dependencies
------------
You need to have OpenCL installed on your machine as well as any drivers
required by your OpenCL backend (ex, nvidia GPU drivers). Further, you'll need
bazel and a modern compiler supporting C++17.

Additionally, you'll need this OpenCL utilities library I wrote:
https://github.com/jsharf/clutil.

Then, place clutil/ and this repo within the same parent directory and create a
Bazel WORKSPACE file in the parent directory.

Then, you should be able to run `bazel run path/to/nnet:nnet_test` to verify that unit
tests pass.


Example Code
------------

One of the goals of the framework is to expose machine learning in as simple of
an interface as possible. For example,
[cifar_test](https://github.com/jsharf/math/blob/master/nnet/cifar_test.cc)
declares a deep convolutional neural network with this easily readable
high-level code: 

```C++
  nnet::Architecture model(kInputSize);
  model
      .AddConvolutionLayer(
          {
              32, // width
              32, // height
              3,  // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              3,  // filter z depth size.
              1,  // stride.
              2,  // padding.
              16, // number of filters.
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{32, 32, 16},
          /* Output size */ nnet::AreaDimensions{16, 16})
      .AddConvolutionLayer(
          {
              16, // width
              16, // height
              16, // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              16, // filter z depth size.
              1,  // stride.
              2,  // padding.
              20, // number of filters.
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{16, 16, 20},
          /* output size */ nnet::AreaDimensions{8, 8})
      .AddConvolutionLayer(
          {
              8,  // width
              8,  // height
              20, // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              20, // filter z depth size.
              1,  // stride.
              2,  // padding.
              20, // number of filters.
          })
      .AddMaxPoolLayer(/* Input size */ {8, 8, 20},
                       /* output size */ {4, 4})
      // No activation function, the next layer is softmax which functions as an
      // activation function
      .AddDenseLayer(10, symbolic::Identity)
      .AddSoftmaxLayer(10);
  std::cout << "Initializing network..." << std::endl;
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::CrossEntropy);
```

Building
--------
Plasticity uses [Bazel](https://bazel.build/) as its build system. I personally
recommend downloading [Bazelisk](https://github.com/bazelbuild/bazelisk) and using that to manage the version of Bazel on your machine. It takes some extra work, but manages Bazel updates for you so that you can always be up to date. Also, it's compatible with  .bazelversion files, so that projects can specify an exact version of Bazel to use for the build (to solve BUILD compatibility issues).

You can build and run the unit tests with bazel build nnet:nnet_tests.

Example Applications
--------------------

For machine learning and neural networks, the main file to import is nnet.h. This is included in target //nnet:nnet. If you're looking for example usage code, check out cifar_test.cc or the unit tests at nnet_test.cc

