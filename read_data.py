# import pandas as pd
# from pandas import read_csv
# import time


# class ReadData:
#     _url = "./rdata.csv"
#     _names = [
#         'Title',
#         'Political_Lean',
#         'Score',
#         'ID',
#         'Subreddit',
#         'URL',
#         'Num_Of_Comments',
#         'Text',
#         'Data_Created',
#     ]

#     def __init__(self):
#         self.data = []

#     def _data_reader(self):
#         dataset = read_csv(self._url, names=self._names).values
#         df = pd.DataFrame(dataset).iloc[:, (0)]
#         return df

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
