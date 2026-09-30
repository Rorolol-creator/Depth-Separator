# Depth-Separator

## Overview

This project ported a C++ depth-separator algorithm into CUDA.

## Test

```sh
cmake -S . -B build --preset release -D USE_CUDA=ON
cmake --build build

export GST_PLUGIN_PATH=$(pwd)
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so
```

Launch gst:

```sh
gst-launch-1.0 uridecodebin uri=file://$(pwd)/TODO_VIDEO ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4
```

or:

```sh
gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```
or:

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

or:

```sh
gst-launch-1.0 uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4
```

To see the fps:

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```
