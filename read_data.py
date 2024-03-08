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
        self.counter = 0
        self.count = 15

    def __iter__(self):
        return self
    
    def data_reader(self):
        dataset = read_csv(self.url, names=self.names).values
        df = pd.DataFrame(dataset).iloc[:, (0)]
        return df

    def counter_func(self, count):

        for i in range(self.count):
            self.counter += 1
            if self.counter >= self.count:
                print('End of Training Run')
                break
            