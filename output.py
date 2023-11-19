import time
import sys
from usercontrol import *
from PIL import Image

def basic_progress_bar(iterable,  prefix='', suffix='', length=40, fill='=', print_end='\r'):
    """
    Display a styled progress bar in the console.
    """

    total = len(iterable)

    def print_bar(iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * (filled_length-1) + '>' + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix}[{bar}] {percent}% {suffix}{print_end}')
        sys.stdout.flush()

    for i, item in enumerate(iterable, 1):
        yield item
        print_bar(i)

    sys.stdout.write('\n')
    sys.stdout.flush()

def load_instructions():

    img = Image.open("instructions.png")
    img = img.convert('L')
    img_array = np.array(img)

    return img_array


def preview_by_matplotlib(
        image_frame_func,
        controllable_vars,
        fps=30,
        timing=False,
        console_input = False):


    def input_callback(key_press: str):

        if key_press >= 'A' and key_press <= 'Z':
            key_press = key_press.lower()

        if key_press in user_control_keymap:
            user_control_keymap[key_press](controllable_vars)

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # overwrite the default keymap
    rcParams['keymap.save'] = {}
    rcParams['keymap.quit'] = {}

    plt.ion()
    figure, ax = plt.subplots()

    # display instructions, press any key to continue
    ax.imshow(load_instructions(), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Press any key to continue')
    figure.canvas.draw()
    figure.canvas.flush_events()

    plt.waitforbuttonpress()

    ax.set_title('Press Enter to Exit')

    # draw figure
    figure.canvas.mpl_connect(
        'key_press_event', lambda event: input_callback(event.key))
    
    frames_index_array = range(0, int(1e9))

    for frame_idx in frames_index_array:

        if controllable_vars.preview_ends:
            break

        if timing:
            t1 = time.time()
        plt.imshow(image_frame_func(frame_idx), cmap='gray', vmin=0, vmax=1)
        figure.canvas.draw()
        figure.canvas.flush_events()
        if timing:
            t2 = time.time()
            print("frame: ", frame_idx, "time: ", t2 - t1)
        time.sleep(1/fps)


def preview_by_opencv(
        image_frame_func,
        controllable_vars,
        fps=30,
        timing=False):

    def input_callback(key_press):
        if key_press == -1:
            return

        if key_press >= ord('A') and key_press <= ord('Z'):
            key_press = key_press + ord('a') - ord('A')

        if key_press == 13:
            key_press = 'enter'
        else:
            key_press = chr(key_press)

        if key_press in user_control_keymap:
            user_control_keymap[key_press](controllable_vars)

    import cv2

    # get the size of first frame to determine the size of the window
    frames_index_array = range(0, int(1e9))
    height, width = image_frame_func(0).shape
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

    # display instructions, press any key to continue
    instructions = cv2.resize(
        load_instructions(), (250, 250), interpolation=cv2.INTER_NEAREST)
    cv2.setWindowTitle('preview', 'Press any key to continue')
    cv2.imshow("preview", instructions)
    cv2.waitKey(0)
    cv2.setWindowTitle('preview', 'Press Enter to Exit')

    # draw figure

    for frame_idx in frames_index_array:

        if controllable_vars.preview_ends:
            break

        if timing:
            t1 = time.time()

        pixel_matrix = image_frame_func(frame_idx)
        resized_pixel_matrix = cv2.resize(
            pixel_matrix, (width*5, height*5), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("preview", resized_pixel_matrix)
        key = cv2.waitKey(int(1000/fps))
        if input_callback is not None:
            input_callback(key)

        if timing:
            t2 = time.time()
            print("frame: ", frame_idx, "time: ", t2 - t1)

    cv2.destroyAllWindows()


def render_by_matplotlib(
        image_frame_func,
        video_length,
        fps=30,
        processes=1,
        output_video_name='output_video',
        output_format='gif'):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if processes > 1:
        from common.joblib.joblib import Parallel, delayed
        from common.joblib_progress.joblib_progress import joblib_progress

    # to make the gif loop, make the video length interger multiple of 200
    number_of_frames = int(np.ceil(video_length * fps / 200) * 200)

    fig = plt.figure()

    frames_index_array = range(0, number_of_frames)
    # TODO: remove this debug line
    # frames_index_array = [4]

    if processes > 1:
        with joblib_progress("Rendering frames...", total=number_of_frames):
            rendered_frames = Parallel(n_jobs=processes)(delayed(image_frame_func)(frame_idx)
                                                         for frame_idx in frames_index_array)
            frames = []
            for frame_idx in frames_index_array:
                frames.append([plt.imshow(rendered_frames[frame_idx],
                              cmap='gray',  vmin=0, vmax=1, animated=True)])
    else:
        frames = []
        for frame_idx in basic_progress_bar(frames_index_array, prefix='Render Progress:', suffix='Complete', length=20):
            frames.append([plt.imshow(image_frame_func(
                frame_idx), cmap='gray',  vmin=0, vmax=1, animated=True)])

    print("Writing to file...")

    ani = animation.ArtistAnimation(fig, frames, interval=1000/fps, blit=True)
    if output_format == 'gif':
        ani.save(output_video_name + ".gif", writer='pillow', fps=fps)
    else:
        ani.save(output_video_name + '.' +
                 output_format, writer='ffmpeg', fps=fps)
