import common.keyboard.keyboard as keyboard
import threading

def console_listen_for_keypress(callback_func, exit_flag):

    """
    spyder does not support interactive matplotlib, so
    we have to take input from console 
    """
    while not exit_flag.is_set():   

        print(keyboard.read_key())


def block_until_keypress():
    exit_flag = threading.Event()
    console_listen_for_keypress(lambda key: exit_flag.set(), exit_flag)

console_listen_for_keypress(lambda key: print(key), threading.Event())