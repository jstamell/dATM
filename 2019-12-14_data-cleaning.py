### This file creates the data sets for the project ###

#%%
import os
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.special import digamma, loggamma
import seaborn as sns
import matplotlib as plt
import datetime
from scipy import sparse
import sys
import gc
import sqlite3
import json

gc.collect()

#%%
os.chdir('C:\\Users\\jakes\\Documents\\Programming_projects\\CU_F19\\graph\\Project')

#%% Setting up arrays to read in training data
f = open('./data/mxm_dataset_train.txt','r')

word_index = np.empty(shape=5000)
track_index = pd.DataFrame(columns=['track_id','mxm_track_id'])
word_list = []

# %% Parsing training data
print(datetime.datetime.now().time())
counter = 0
for i in f:
    if i[0] == '#':
        pass
    elif i[0] == '%':
        word_index = i[1:].split(',')
    else:
        contents = i.split(',')
        track_index = track_index.append({'track_id':contents[0],'mxm_track_id':contents[1]}, ignore_index=True)
        word_count_dict = {}
        for j in contents[2:]:
            key,value = j.split(':')
            word_count_dict[int(key)] = int(value)    
        word_list.append(word_count_dict)
        counter += 1
    print(counter)
#    if counter > 1000:
#        break
print(datetime.datetime.now().time())
f.close()

#%% Converting training data to sparse matrix format
words_sparse = sparse.dok_matrix((len(word_list),5000), dtype=np.uint8)

for i in range(len(word_list)):
    for j in word_list[i]:
        words_sparse[i,j-1] = word_list[i][j]

#%% Saving if needed
words_sparse = words_sparse.tocsr()
sparse.save_npz('./data/word-counts-sparse.npz',words_sparse)

#%% Inserting tracks df into db and then pulling out the metadata on these tracks

conn_tmdb = sqlite3.connect('./data/track_metadata.db')

conn_tmdb.execute('''CREATE TABLE IF NOT EXISTS lyr_tracks
                  (track_id VARCHAR)''')

df = track_index.iloc[:,[0]]
df.columns = sqlite3.get_column_names_from_db_table(conn_tmdb, 'lyr_tracks')
df.to_sql(name='lyr_tracks', con=conn_tmdb, if_exists='append', index=False)

qu = '''SELECT track_id, title, song_id, artist_id, artist_name, year 
                        FROM songs
                        WHERE track_id in (SELECT track_id from lyr_tracks)'''

track_info2 = pd.read_sql_query(qu, conn_tmdb)

conn_tmdb.close()

#%% Saving intermediate tables

with open('./data/word-counts-list', 'w') as fout:
    json.dump(word_list, fout)
    
track_index.to_csv('./data/train-track-index.csv')
track_info2.to_csv('./data/train-track-meta.csv')
np.save('./data/word-index.npy',word_index)

##% Doing the same for the test data

#%% Setting up arrays to read in training data
f = open('./data/mxm_dataset_test.txt','r')

test_index = pd.DataFrame(columns=['track_id','mxm_track_id'])
word_list = []

# %% Parsing training data
print(datetime.datetime.now().time())
counter = 0
for i in f:
    if i[0] == '#':
        pass
    elif i[0] == '%':
        pass
    else:
        contents = i.split(',')
        test_index = test_index.append({'track_id':contents[0],'mxm_track_id':contents[1]}, ignore_index=True)
        word_count_dict = {}
        for j in contents[2:]:
            key,value = j.split(':')
            word_count_dict[int(key)] = int(value)    
        word_list.append(word_count_dict)
        counter += 1
    print(counter)
#    if counter > 1000:
#        break
print(datetime.datetime.now().time())
f.close()

#%% Converting training data to sparse matrix format
test_sparse = sparse.dok_matrix((len(word_list),5000), dtype=np.uint8)

for i in range(len(word_list)):
    for j in word_list[i]:
        test_sparse[i,j-1] = word_list[i][j]

#%% Saving if needed
test_sparse = test_sparse.tocsr()
sparse.save_npz('./data/test-counts-sparse.npz',test_sparse)

#%% Inserting tracks df into db and then pulling out the metadata on these tracks

conn_tmdb = sqlite3.connect('./data/track_metadata.db')

conn_tmdb.execute('''CREATE TABLE IF NOT EXISTS test_tracks
                  (track_id VARCHAR)''')

df = test_index.iloc[:,[0]]
#df.columns = sqlite3.get_column_names_from_db_table(conn_tmdb, 'test_tracks')
df.to_sql(name='test_tracks', con=conn_tmdb, if_exists='append', index=False)

qu = '''SELECT track_id, title, song_id, artist_id, artist_name, year 
                        FROM songs
                        WHERE track_id in (SELECT track_id from test_tracks)'''

test_info2 = pd.read_sql_query(qu, conn_tmdb)

conn_tmdb.close()

#%% Saving intermediate tables

with open('./data/test-counts-list', 'w') as fout:
    json.dump(word_list, fout)
    
test_index.to_csv('./data/test-track-index.csv')
test_info2.to_csv('./data/test-track-meta.csv')