# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:33:07 2018

@author: Yung
"""

import os
os.chdir(r'C:\Users\yung\Desktop\NP methodology\author_network')

import networkx as nx
import pandas as pd
import itertools

df = pd.read_excel('190127_netpharm_list(2017_affiliation).xlsx', sheetname=0, index_col='약물')
df_T = df.T

## Author network 구축


G  = nx.Graph()

# co-authorship network construction 

for i in range(len(df_T.index)):
    
#    corr_author = df_T['Corr_author'].iloc[i].split(",")
    auth = df_T['Author'].iloc[i].split(",")
    year = df_T['Year'].iloc[i]
    
    for j in itertools.combinations(auth,2):
        G.add_edge(*j, year=year)
        
G_edge = nx.to_pandas_edgelist(G)
G_edge.to_excel('author_edge_list.xlsx')

G_node = pd.DataFrame(list(G.nodes()))
#G_node.to_excel('author_node_list.xlsx')

degree_auth = pd.DataFrame(list(nx.degree(G)), columns=['index','degree'])
degree_auth.to_excel('author_degree.xlsx')
        
#G_nodes_frame = pd.DataFrame(G.nodes())
#G_nodes_frame.to_excel('authors_list.xlsx') 
#   
#G_dic_frame = pd.DataFrame(G.edges())
#G_dic_frame.to_excel('author_network.xlsx') 
#    
    



## Affliation network construction

G_aff = nx.Graph()

# co-authorship network construction 

for i in range(len(df_T.index)):
    
#    corr_author = df_T['Corr_author'].iloc[i].split(",")
    auth_aff = df_T['Author'].iloc[i].split(",")
    year = df_T['Year'].iloc[i]
    
    for j in itertools.combinations(auth_aff,2):
        G_aff.add_edge(*j, year=year)
        
G_edge_aff = nx.to_pandas_edgelist(G_aff)
G_edge_aff.to_excel('affliation_edge_list.xlsx')

G_node_aff = pd.DataFrame(list(G_aff.nodes()))
#G_node_aff.to_excel('affliation_node_list.xlsx')

degree_aff = pd.DataFrame(list(nx.degree(G_aff)), columns=['index','degree'])
degree_aff.to_excel('affliation_degree.xlsx')



    