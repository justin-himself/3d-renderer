import concurrent.futures
import queue
import time

# Function to calculate the Fibonacci series


def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Function to calculate and put the result into the priority queue


def calculate_and_enqueue(queue, n):
    result = fibonacci(n)
    queue.put((n, result))

# Function to pop from the queue and print the result


def dequeue_and_print(queue):
    while True:
        time.sleep(0.15)
        try:
            n, result = queue.get(block=False)
            print(f"Fibonacci({n}) = {result}")
            queue.task_done()
        except queue.Empty:
            print("empty")
            pass


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        priority_queue = queue.PriorityQueue(maxsize=100)

        # Create a single thread for dequeuing and printing results
        dequeue_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        dequeue_thread.submit(dequeue_and_print, priority_queue)

        # Continuously submit Fibonacci calculations in real-time
        n = 1
        while True:
            executor.submit(calculate_and_enqueue, priority_queue, n)
            n += 1
            # Sleep for a short duration before the next calculation
            time.sleep(0.1)
            print(priority_queue.qsize())
