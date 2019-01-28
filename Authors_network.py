# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:33:07 2018

@author: Yung
"""

import os
os.chdir(r'C:\Users\Yung\Desktop\netpharm comparison\author_network')

import networkx as nx
import pandas as pd
import networkx as nx
import itertools

df = pd.read_excel('180629_netpharm_list(2017 update).xlsx', sheetname=0, index_col='약물')
lists = df.T

G  = nx.Graph()

for i in lists.Author:
    auth = i.split(",")
    for j in itertools.combinations(auth,2):
        G.add_edge(*j)
        

G_edge = nx.to_pandas_edgelist(G)
G_edge.to_excel('author_edge_list.xlsx')

G_node = pd.DataFrame(list(G.nodes()))
G_node.to_excel('author_list.xlsx')
        
#G_nodes_frame = pd.DataFrame(G.nodes())
#G_nodes_frame.to_excel('authors_list.xlsx') 
#   
#G_dic_frame = pd.DataFrame(G.edges())
#G_dic_frame.to_excel('author_network.xlsx') 
#    
    
    