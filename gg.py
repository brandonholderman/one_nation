def counter_func(func_run):
    counter = 0
    arr = []

    for i in range(func_run):
        counter += 1
        # print(counter)
        arr.append(counter)
        # print(arr[i])
        if counter >= 15:
            break
    
    # print(arr)






counter_func(30)