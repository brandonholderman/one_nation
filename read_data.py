import pandas as pd
from pandas import read_csv


class ReadData:
        
    url = "./rdata.csv"
    names = [
        'Title',
        'Political_Lean',
        'Score',
        'ID',
        'Subreddit',
        'URL',
        'Num_Of_Comments',
        'Text',
        'Data_Created',
    ]

    def __init__(self):
        self.data = []

    def __iter__(self):
        return self
    
    def data_reader(self):
        dataset = read_csv(self.url, names=self.names).values
        df = pd.DataFrame(dataset).iloc[:, (0)]
        return df

    def counter_func(self, count):
        counter = 0
        for i in range(count):
            if counter >= count:
                print(f'{counter} + End of Training Run')
                break
            else:
                counter += 1
            

# def counter_func(func_run):
#     counter = 0
#     arr = []

#     for i in range(func_run):
#         counter += 1
#         # print(counter)
#         arr.append(counter)
#         # print(arr[i])
#         if counter >= 15:
#             break