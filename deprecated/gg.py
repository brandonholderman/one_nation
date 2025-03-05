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



    # def __call__(self):
    #     self.count += 1
    #     return count

    # def counter_func(self, count):
    #     counter = self.counter

    #     for i in range(count):
    #         if counter >= count:
    #             print(f'{counter} - Conditional Hit, End of Training Run')
    #             break
    #         else:
    #             counter += 1
    #             print(f'{counter} - Increase counter function ran')
    #             return
