import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

chunk_num = 0
with open('data/test_list.csv', 'w') as fout:
    fout.write("user_id,anime_id,rating,watching_status\n")
    with pd.read_csv('data/filtered_list.csv',sep=',',chunksize=10_000_000, usecols=['user_id', 'anime_id', 'rating', 'watching_status']) as reader:
        for chunk in reader:
            print("Currently on chunk ", chunk_num)
            assert isinstance(chunk, DataFrame)
            chunk = chunk.query('user_id <= 1200')
            chunk.to_csv(fout,header=False,index=False, line_terminator='\n')
            chunk_num += 1