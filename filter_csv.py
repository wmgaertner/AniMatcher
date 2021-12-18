import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

# remove entries from userlists that the user has not given a score or not completed/dropped

chunk_num = 0
with open('data/filtered_list.csv', 'w') as fout:
    fout.write("user_id,anime_id,rating,watching_status\n")
    with pd.read_csv('data/animelist.csv',sep=',',chunksize=10_000_000, usecols=['user_id', 'anime_id', 'rating', 'watching_status']) as reader:
        for chunk in reader:
            print("Currently on chunk ", chunk_num)
            assert isinstance(chunk, DataFrame)
            chunk = chunk.query('rating != 0 and (watching_status == 2 or watching_status == 4)')
            chunk.to_csv(fout,header=False,index=False, line_terminator='\n')
            chunk_num += 1



