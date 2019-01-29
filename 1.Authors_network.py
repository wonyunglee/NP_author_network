# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:33:07 2018

@author: Yung
"""

import os
os.chdir(r'C:\Users\yung\Desktop\NP methodology\author_network')

import numpy as np
import networkx as nx
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

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
    auth_aff = df_T['affiliation'].iloc[i].split(",")
    year = df_T['Year'].iloc[i]
    
    for j in itertools.combinations(auth_aff,2):
        G_aff.add_edge(*j, year=year)
        
G_edge_aff = nx.to_pandas_edgelist(G_aff)

G_edge_aff.to_excel('affliation_edge_list.xlsx')

G_node_aff = pd.DataFrame(list(G_aff.nodes()))
#G_node_aff.to_excel('affliation_node_list.xlsx')

degree_aff = pd.DataFrame(list(nx.degree(G_aff)), columns=['index','degree'])
degree_aff.to_excel('affliation_degree.xlsx')


## affiliation, paper by year

aff_paper_year = pd.DataFrame(np.array(np.zeros((8,2))),columns = (['year','affiliation']), 
                              index = ([i for i in range(2011,2019)]))

aff = pd.DataFrame(df_T[['affiliation','Year']])

for i in range(2011,2019):
    
    author_group = (aff['Year'] == i)
    author_group = aff[author_group]
    
    author_list = []
    
    for j in range(len(author_group.index)):        
        author_list.extend(author_group.iloc[j])
        
    author_list = list(set(author_list))        
    aff_paper_year['affiliation'].loc[i] = len(author_list)    
#        
#aff_1 = pd.DataFrame(G_edge_aff[['source', 'year']])
#aff_1['1'] = np.array(np.ones((len(aff_1.index))))
#pivot_1 = aff_1.pivot_table(index='year', values='1', aggfunc='sum' )
#
#aff_2 = pd.DataFrame(G_edge_aff[['target', 'year']])
#aff_2['1'] = np.array(np.ones((len(aff_1.index))))
#pivot_2 = aff_2.pivot_table(index='year', values='1', aggfunc='sum')
#
#aff_pivot = pivot_1 + pivot_2
#
#aff_paper_year['affiliation'] = aff_pivot['1']


paper =  pd.DataFrame(df_T['Year'])
paper['1'] = np.array(np.ones((len(paper.index))))
pivot_3 = paper.pivot_table(index='Year',values='1', aggfunc='sum')

aff_paper_year['year'] = pivot_3['1']

# visualization

plt.plot(aff_paper_year['year'], 'r')
plt.plot(aff_paper_year['affiliation'], 'b')
plt.legend(['Affiliation', 'paper'])
plt.show()




### drug-availability(DA) and drug-target interaction (DTI) ratio visualization

df_method = pd.read_excel('190127_netpharm_list(2017_affiliation).xlsx', sheetname='drug_availavility,DTIs_methods')

# DA visualization

df_DA = df_method[['year','DTIs construction','Drug availability','OBDL']]
df_DA_ratio = pd.DataFrame(np.array(np.zeros((8,2))),columns = (['DA','OB_DL']), 
                              index = ([i for i in range(2011,2019)]))
for i in range(2011,2019):
    
    DA_method = (df_DA['year'] == i)
    DA_method = df_DA[DA_method]
    
    ratio_DA = DA_method['Drug availability'].sum() / DA_method['DTIs construction'].sum()
    ratio_OBDL = DA_method['OBDL'].sum() / DA_method['DTIs construction'].sum()
    
    df_DA_ratio.loc[i] = np.array([[ratio_DA,ratio_OBDL]])
    
plt.plot(df_DA_ratio['DA'], 'coral')
plt.plot(df_DA_ratio['OB_DL'], 'r')
plt.legend(['Drug availability', 'OB & DL'])
plt.show()

df_DA_ratio.to_excel('DA_ratio.xlsx')


# DTI visualization

df_DTI = df_method[['year','Similarity-based approach','Structure-based approach','Experiment-based approach']]
df_DTI_ratio = pd.DataFrame(np.array(np.zeros((8,3))),columns = (['Similarity-based approach','Structure-based approach','Experiment-based approach']), 
                              index = ([i for i in range(2011,2019)]))

for j in range(2011,2019):
    
    DTI_method = (df_DTI['year'] == j)
    DTI_method = df_DTI[DTI_method]
    
    DTI_sum = DTI_method[['Similarity-based approach','Structure-based approach','Experiment-based approach']]
    DTI_sum = DTI_sum.sum().sum()
    
    ratio_similrity = DTI_method['Similarity-based approach'].sum() / DTI_sum
    ratio_structure = DTI_method['Structure-based approach'].sum() / DTI_sum
    ratio_experiment = DTI_method['Experiment-based approach'].sum() / DTI_sum

    df_DTI_ratio.loc[j] = np.array([[ratio_similrity,ratio_structure,ratio_experiment]])


red = np.array([229,36,38]) / 256
green = np.array([243,232,0])/256
blue = np.array([32,114,178])/256

plt.plot(df_DTI_ratio['Similarity-based approach'], color= red)
plt.plot(df_DTI_ratio['Structure-based approach'], color= green)
plt.plot(df_DTI_ratio['Experiment-based approach'], color= blue)
plt.legend(['Similarity-based approach','Structure-based approach','Experiment-based approach'])
plt.show()

    
df_DTI_ratio.to_excel('DTI_ratio.xlsx')

    