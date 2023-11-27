# 3d Renderer

> This project is licensed under the MIT LICENSE.


## Quick start 

### Setup

First, clone the latest source code from github.

```bash
git clone --recursive https://github.com/justin-himself/3d-renderer.git
```

You only need to have `numpy` and `matplotlib` installed to run the program, but it is recommanded to install `opencv`, `joblib` and `joblib-progress` to achieve best performance. You will also need to install `ffmpeg` if you want the output to be any format other than `.gif`, though it's entirely optional.

Simply execute the following command to install all requirements.

```bash
pip install -r requirements.txt
```

It is possible to run the program in Spyder but interactive preview, multicore rendering and custom output format are **NOT** supported, it's hugely recommended that you run the program directly in the console.

### Render a Toilet

To render a spinning toilet, use 

```bash
python3 main.py models/toilet_1451.obj output
```

A window will pop up so you can adjust the object's positon. Press enter when you are satisfied and the render will start.

The result will be saved as `output.gif`

## Models & Rendering

Currently the program only supports rendering low poly wavefront (obj) files. The components must be joined together.

To genenerate such models, follow steps below

1. Download the model from internet
2. Import it into blender
3. Select all parts and rightclick, click "join"
4. Resize and align the model to center of all axis
5. (Optional) Add a "Decimate" modifier if your model have too much polygons
6. Export as wavefront file and in the pop up menu, deselect every option (eg. normals) except for "Apply modifiers"

This project uses multiprocessor cpu rendering. If you have a NVIDIA GPU, you can also download [Cupy](https://cupy.dev/) and replace `import numpy` to `import cupy`, which should give a decent performance boost. I was not able to test this out since I use an Apple Silicon Mac to develop the project.

## Demo 

![bomb](https://github.com/justin-himself/3d-renderer/blob/main/demo/bomb.gif?raw=true)

For other demos, including a car and a spinning toilet, go to  
https://github.com/justin-himself/3d-renderer/tree/main/demo