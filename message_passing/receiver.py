import sysv_ipc
import numpy as np
import struct
import threading  # Import the threading module

BUFF_SIZE = 16

from type_definitions import *

SIZEOF_FLOAT = 8

lock = threading.Lock()


# Define a function that contains the existing code
def process_messages():
    try:
        mq = sysv_ipc.MessageQueue(1234, sysv_ipc.IPC_CREAT)

        global p_messsage

        with lock:
            p_messsage = "hello"

        while True:
            message, mtype = mq.receive()
            print("*** New message received ***")
            print(f"Raw message: {message}")
            if mtype == TYPE_STRING:
                str_message = message.decode()
                print(f"Interpret as string: {str_message}")
                with lock:
                    p_messsage = str_message

            elif mtype == TYPE_TWODOUBLES:
                two_doubles = struct.unpack("dd", message)
                print(f"Interpret as two doubles: {two_doubles}")

            elif mtype == TYPE_NUMPY:
                numpy_message = np.frombuffer(message, dtype=np.int8)
                print(f"Interpret as numpy: {numpy_message}")

            elif mtype == TYPE_DOUBLEANDNUMPY:
                one_double = struct.unpack("d", message[:SIZEOF_FLOAT])[0]
                numpy_message = np.frombuffer(message[SIZEOF_FLOAT:], dtype=np.int8)
                print(f"Interpret as double and numpy: {one_double}, {numpy_message}")

    except sysv_ipc.ExistentialError:
        print("ERROR: message queue creation failed")


# Define a function for the main thread to print "Hello" continuously
def print_hello():
    while True:
        with lock:
            print(p_messsage)


if __name__ == "__main__":
    # Create a thread to execute the process_messages function
    message_processing_thread = threading.Thread(target=process_messages)

    # Start the message processing thread
    message_processing_thread.start()

    # Start the main thread to print "Hello" continuously
    print_hello()
