# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:53:22 2023

@author: JChuj
"""
import pandas as pd
import json
import torch
import torch.nn as nn


edge_df = pd.read_csv('edges.csv')
node_df = pd.read_csv('nodes.csv')
node_df = node_df.drop_duplicates(['spotify_id']).reset_index(drop=True)
edge_df = edge_df.drop_duplicates().reset_index(drop=True)
int_id_dict, genre_id_dict, chart_hit_dict = dict(), dict(), dict()
for i in range(len(node_df)):
    row_id = node_df.loc[i]['spotify_id']
    if row_id not in int_id_dict.keys():
        int_id_dict[row_id] = i
        k = node_df.loc[i]['genres'].strip('][').split(', ')
        for genre in k:
            if genre not in genre_id_dict.keys():
                genre_id_dict[genre] = len(genre_id_dict.keys())
        j = str(node_df.loc[i]['chart_hits']).strip('][').split(', ')
        for hit in j:
            if hit == 'nan':
                continue
            country = hit[1:3]
            if country not in chart_hit_dict.keys():
                chart_hit_dict[country] = len(chart_hit_dict.keys())
    else:
        print(row_id, ' ', i, ' ', int_id_dict[row_id])


int_dict = open('int_id.json', 'w')
genre_dict = open('genre_id.json', 'w')
chart_dict = open('chart_is.json', 'w')
int_json = json.dumps(int_id_dict)
int_dict.write(int_json)
int_dict.close()
genre_json = json.dumps(genre_id_dict)
genre_dict.write(genre_json)
genre_dict.close()
chart_json = json.dumps(chart_hit_dict)
chart_dict.write(chart_json)
chart_dict.close()
