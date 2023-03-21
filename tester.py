# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:53:22 2023

@author: JChuj
"""
import pandas as pd
import json
import torch
import torch.nn as nn
from bert_encoder import encode_text

def generate_embeddings():
    edge_df = pd.read_csv('edges.csv')
    node_df = pd.read_csv('nodes.csv')
    node_df = node_df.drop_duplicates(['spotify_id']).reset_index(drop=True)
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    
    with open('int_id.json') as int_json:
        int_id_dict = json.load(int_json)
    with open('genre_id.json') as genre_json:
        genre_id_dict = json.load(genre_json)
    with open('chart_is.json') as chart_json:
        chart_hit_dict = json.load(chart_json)
    
    genres = []
    for x in genre_id_dict:
        genres.append(x.strip("'").strip('\"'))
        
    encoding_dict = {}
    for x in genres:
        encoding_dict[x] = encode_text(x)
        
    def hit_to_int(hit):
        ints = ''
        for i in hit:
            if i.isdigit():
                ints += i
        return int(ints)
    
    nodes = []
    for i in range(len(node_df)):
        node = node_df.loc[i]
        followers = node['followers']
        if followers is None:
            followers = 0
        pop = node['popularity']
        if pop is None:
            pop = 0
        follower_tensor = torch.Tensor([followers])
        pop_tensor = torch.Tensor([pop])
        genre_tensor = torch.zeros_like(torch.Tensor(768))
        hit_tensor = torch.zeros_like(torch.Tensor(71))
        genres = node['genres'].strip("][").split(', ')
        for genre in genres:
            genre = genre.strip("'").strip('\"')
            genre_tensor += encoding_dict[genre].squeeze()
        hits = str(node_df.loc[i]['chart_hits']).strip('][').split(', ')
        for hit in hits:
            if hit == 'nan':
                continue
            country = hit[1:3]
            hit_tensor[chart_hit_dict[country]] = hit_to_int(hit)
        node_tensor = torch.cat((follower_tensor, pop_tensor, genre_tensor, hit_tensor))
        nodes.append(node_tensor.unsqueeze(0))
    
    nodes_tensor = torch.cat(tuple(nodes), dim=0)
    
    edges = []
    for i in range(len(edge_df)):
        edge = edge_df.loc[i]
        if edge['id_0'] in int_id_dict and edge['id_1'] in int_id_dict:
            edges.append([int_id_dict[edge['id_0']], int_id_dict[edge['id_1']]])
            edges.append([int_id_dict[edge['id_1']], int_id_dict[edge['id_0']]])
    edges = torch.Tensor(edges)
    return nodes_tensor, edges

nodes, edges = generate_embeddings()
    
'''
word_freq = dict()
word_id = dict()
for i in range(len(node_df)):
        k = node_df.loc[i]['genres'].strip("][").split(', ')
        for genre in k:
            genre = genre.strip("''")
            for word in genre:
                if word not in genre_id_freq.keys():
'''

'''
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

        
for i in range(len(edge_df)):
    x, y = edge_df.loc[i]['id_0'], edge_df.loc[i]['id_1']
    if x in int_id_dict.keys() and y in int_id_dict.keys():
        edge_df.loc[i]['int_id0'] = int_id_dict[edge_df.loc[i]['id_0']]
        edge_df.loc[i]['int_id1'] = int_id_dict[edge_df.loc[i]['id_1']]
    else: 
        edge_df.drop(i, axis='index')

edge_df = edge_df.reset_index(drop=True)
'''