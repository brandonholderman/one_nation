import time

def counter_func(func_run):
    counter = 0
    arr = []

    for i in range(func_run):
        counter += 1
        # print(counter)
        arr.append(counter)
        print(counter)
        time.sleep(1)
        if counter >= func_run:
            break
    # print(arr)

counter_func(3)