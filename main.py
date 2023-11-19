import argparse
from multiprocessing import cpu_count
from obj_ops import load_objfile
from output import preview_by_opencv, render_by_matplotlib, preview_by_matplotlib
from animation import rotation_animation
from usercontrol import ControlableVars


def env_validate(args):

    try:
        import numpy
        import matplotlib
    except ImportError:
        print("Numpy & Matplotlib not found, exiting")
        exit(-1)

    if args.processors > 1:
        try:
            import joblib
            import joblib_progress
        except ImportError:
            print(
                f"--processors {args.processors}: joblib & joblib_progress not found, using single processor")
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


def main(args):

    args = env_validate(args)

    # receive the args
    model_filepath = args.model
    fps = args.fps
    render_wireframe = args.render_wireframe
    render_filled = args.render_filled
    preview_method = args.preview_method
    processors = args.processors
    output_length = args.output_length
    output_width = args.output_width
    output_height = args.output_height
    output_format = args.output_format
    output_file = args.output_file

    if preview_method == 'opencv':
        preview_func = preview_by_opencv
    elif preview_method == 'matplotlib':
        preview_func = preview_by_matplotlib
    elif preview_method == 'none':
        preview_func = lambda *args, **kwargs: None

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

    # select one
    render_group = parser.add_mutually_exclusive_group(required=False)
    render_group.add_argument('--render-wireframe', action='store_true',
                              help='Render the model in wireframe mode')

    render_group.add_argument('--render-filled', action='store_true', default=True,
                              help='Render the model in filled mode')

    # Optional arguments
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

    main(args)
