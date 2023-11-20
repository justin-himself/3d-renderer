import argparse
from multiprocessing import cpu_count
from obj_ops import load_objfile
from output import preview_by_opencv, render_by_matplotlib, preview_by_matplotlib
from animation import rotation_animation
from usercontrol import ControlableVars


def is_spyder():
    """
    detect specifically if the script is running in Spyder
    """

    try:
        # Check if 'get_ipython' function is defined
        get_ipython
        exec_lines = get_ipython().config['IPKernelApp']['exec_lines']
        for line in exec_lines:
            if 'spyder' in line:
                return True
        return False
    except NameError:
        return False


def spyder_interactive_args_input():

    def propmpt_input(prompt, default, type):
        userinput = None
        while userinput is None or type(userinput) != type:
            userinput = input(prompt + f" (default: {default}): ")
            if userinput.strip() == '':
                return default

    args = {}

    # preview will not be availble since spider does not support user interaction
    args['preview_method'] = "none"
    args['processors'] = 1  # assume joblib not available in spyder
    args['output_format'] = "gif"  # assume ffmpeg not available in spyder
    args['output_width'] = 100
    args['output_height'] = 100
    args['output_length'] = 1

    args['model_filepath'] = propmpt_input(
        "Model file path", "models/teapot_158.obj", str)
    args['fps'] = propmpt_input("FPS", 30, int)
    args['output_style'] = propmpt_input("Output style", "filled", str)
    args['output_file'] = propmpt_input("Output file name", "output", str)

    return args


def env_validate(args):

    try:
        import numpy
        import matplotlib
    except ImportError:
        print("Numpy & Matplotlib not found, exiting")
        exit(-1)

    if args.processors > 1:
        try:
            import joblib_progress
            import joblib
        except ImportError:
            print("--processors: joblib not found, fall back to single process")
            args.processors = 1

    if args.preview_method == 'opencv':
        try:
            import cv2
        except ImportError:
            print("--preview-method opencv: opencv not found, fall back to matplotlib")
            args.preview_method = 'matplotlib'

    if args.output_format != 'gif':
        try:
            import ffmpeg
        except ImportError:
            print(
                f"--output-format: {args.output_format}: FFMpeg not found, fall back to gif")
            args.output_format = 'gif'

    return args


def main(
    model_filepath: str,
    fps: int,
    preview_method: str,
    processors: int,
    output_style: str,
    output_length: int,
    output_width: int,
    output_height: int,
    output_format: str,
    output_file: str
):

    if preview_method == 'opencv':
        preview_func = preview_by_opencv
    elif preview_method == 'matplotlib':
        preview_func = preview_by_matplotlib
    elif preview_method == 'none':
        preview_func = lambda *args, **kwargs: None

    if output_style == 'filled':
        render_filled = True
        render_wireframe = False
    elif output_style == 'wireframe':
        render_filled = False
        render_wireframe = True
    elif output_style == 'both':
        render_filled = True
        render_wireframe = True

    origin_mesh = load_objfile(model_filepath)
    controllable_vars = ControlableVars()

    preview_func(
        lambda x: rotation_animation(
            origin_mesh,
            x,
            controllable_vars,
            draw_filled=render_filled,
            draw_wireframe=render_wireframe,
            screen_width=50,
            screen_height=50,
            do_normal_clip=True),
        controllable_vars,
        fps=5
    )
    render_by_matplotlib(
        lambda x: rotation_animation(
            origin_mesh,
            x,
            controllable_vars,
            draw_wireframe=render_wireframe,
            draw_filled=render_filled,
            screen_width=output_width,
            screen_height=output_height,
            do_normal_clip=True,
        ),
        video_length=output_length,
        fps=fps,
        processes=processors,
        output_video_name=output_file,
        output_format=output_format)


if __name__ == "__main__":

    if is_spyder():
        print('''
Warning: Running in Spyder, only basic rendering is supported, 
and performance is heavily bottlenecked.''')
        main(**spyder_interactive_args_input())
        print("Render finished, please check the output file")
        exit(0)

    parser = argparse.ArgumentParser(
        description='3D Model Rendering Script written by Justin')

    parser.add_argument('model', metavar='model_file',
                        help='Wavefront OBJ file to be rendered')

    parser.add_argument('output_file', help='Output file name')

    # Required arguments
    parser.add_argument('--output-length', type=int, default=1,
                        help='Output video length in seconds.')

    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the video')

    # Optional arguments
    parser.add_argument('--output-style', choices=['filled', 'wireframe', 'both'], default='filled',
                        help='Render style')
    parser.add_argument('--preview-method', choices=['opencv', 'matplotlib', 'none'], default='opencv',
                        help='Select the method for preview and render (default: opencv)')
    parser.add_argument('--output-format', default='gif',
                        help='Select the render result format (default: gif)')

    parser.add_argument('--processors', type=int, default=cpu_count(),
                        help='Number of processors used to render the video, requires joblib to work (default: CPU count)')

    parser.add_argument('--output-width', type=int, default=100,
                        help='Width of the output video (default: 100)')

    parser.add_argument('--output-height', type=int, default=100,
                        help='Height of the output video (default: 100)')

    args = parser.parse_args()
    args = env_validate(args)
    main(
        args.model,
        args.fps,
        args.preview_method,
        args.processors,
        args.output_style,
        args.output_length,
        args.output_width,
        args.output_height,
        args.output_format,
        args.output_file
    )
