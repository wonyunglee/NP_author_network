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

sns.set(style="white", context="talk")

df = pd.read_excel('190127_netpharm_list(2017_affiliation).xlsx', sheet_name='drug_availavility,DTIs_methods', index_col='약물')

## H-C, C-T, T-D database, method frequency
# H-C

HC_method = df['HC method']
HC_year = df['year']
HC_method.dropna(axis=0, inplace=True)

HC_fre = pd.DataFrame(columns=['HC_method','Frequency','year'])

i=0

for j in range (len(HC_method.index)):
    
    
    HC = HC_method.iloc[j].split(", ")
    year = HC_year.iloc[j]
    
    if type(HC) == str:
        
        HC_fre.loc[i] = [HC,1,year]
        i += 1
    
    else:
        for p in range(len(HC)):
               
            HC_fre.loc[i] = [HC[p],1,year]
            i += 1

HC_pivot_fre = HC_fre.pivot_table(index='HC_method',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
HC_pivot_fre_year = HC_fre.pivot_table(index=['HC_method','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)

HC_pivot_fre_year = pd.DataFrame(HC_pivot_fre_year.to_records())

fig, ax = plt.subplots()
fig.set_size_inches(12,10)


g = sns.barplot(x=HC_pivot_fre.index, y=HC_pivot_fre['Frequency'])
plt.setp(g.get_xticklabels(), rotation=90)
plt.show()

plt.close()


fig, ax = plt.subplots()
fig.set_size_inches(20,20)
ax.legend(prop={'size':4})


sns.lineplot(data=HC_pivot_fre_year, x='year', y='Frequency', hue='HC_method')
plt.show()

plt.close()





# C-T 

CT_method = df['CT method']
CT_year = df['year']
CT_method.dropna(axis=0, inplace=True)

CT_fre = pd.DataFrame(columns=['CT_method','Frequency','year'])

i=0

for j in range (len(CT_method.index)):
    
    
    CT = CT_method.iloc[j].split(", ")
    year = CT_year.iloc[j]
    
    if type(CT) == str:
        
        CT_fre.loc[i] = [CT,1,year]
        i += 1
    
    else:
        for p in range(len(CT)):
               
            CT_fre.loc[i] = [CT[p],1,year]
            i += 1

CT_pivot_fre = CT_fre.pivot_table(index='CT_method',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
CT_pivot_fre_year = CT_fre.pivot_table(index=['CT_method','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)

CT_pivot_fre_year = pd.DataFrame(CT_pivot_fre_year.to_records())

fig, ax = plt.subplots()
fig.set_size_inches(12,10)



g = sns.barplot(x=CT_pivot_fre.index, y=CT_pivot_fre['Frequency'])
plt.setp(g.get_xticklabels(), rotation=90)

plt.show()
plt.close()


fig, ax = plt.subplots()
fig.set_size_inches(20,20)
ax.legend(prop={'size':4})




sns.lineplot(data=CT_pivot_fre_year, x='year', y='Frequency', hue='CT_method')
plt.show()

plt.close()





# T-D 

TD_database = df['TD database']
TD_year = df['year']
TD_database.dropna(axis=0, inplace=True)

TD_fre = pd.DataFrame(columns=['TD_database','Frequency','year'])

i=0

for j in range (len(TD_database.index)):
    
    
    TD = TD_database.iloc[j].split(", ")
    year = TD_year.iloc[j]
    
    if type(TD) == str:
        
        TD_fre.loc[i] = [TD,1,year]
        i += 1
    
    else:
        for p in range(len(TD)):
               
            TD_fre.loc[i] = [TD[p],1,year]
            i += 1

TD_pivot_fre = TD_fre.pivot_table(index='TD_database',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
TD_pivot_fre_year = TD_fre.pivot_table(index=['TD_database','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)

TD_pivot_fre_year = pd.DataFrame(TD_pivot_fre_year.to_records())

fig, ax = plt.subplots()
fig.set_size_inches(12,10)



g = sns.barplot(x=TD_pivot_fre.index, y=TD_pivot_fre['Frequency'])


plt.setp(g.get_xticklabels(), rotation=90)

plt.show()
plt.close()


fig, ax = plt.subplots()
fig.set_size_inches(20,20)


ax.legend(prop={'size':4})
sns.lineplot(data=TD_pivot_fre_year, x='year', y='Frequency', hue='TD_database')
plt.show()

plt.close()






# CT method by year

CT_year = df[['year','Similarity approach','Structural approach','Experimental approach']]

CT_year_pivot = CT_year.pivot_table(index='year')

fig, ax = plt.subplots()
fig.set_size_inches(12,10)
ax.set_ylabel('Proportion')
sns.lineplot(data=CT_year_pivot)

# Drug availability assessment

DA_year = df[['year','Drug availability','OB, DL']]

DA_year_pivot = DA_year.pivot_table(index='year')


fig, ax = plt.subplots()
fig.set_size_inches(12,10)
ax.set_ylabel('Proportion')
sns.lineplot(data=DA_year_pivot)


### C-T-D network construction

graph = nx.Graph()

# CT_pair

HCT_method = df[['HC method','CT method']]
HCT_method.dropna(axis=0, inplace=True)


for i in range(len(HCT_method.index)):
    HC = HCT_method.iloc[i]['HC method']
    CT = HCT_method.iloc[i]['CT method']
    
    if graph.has_edge(HC, CT):
        graph[HC][CT]['weight'] += 1
        
    else:
        graph.add_edge(HC,CT, weight=1)
    
CTD_method = df[['CT method','TD database']]
CTD_method.dropna(axis=0, inplace=True)

for j in range(len(CTD_method.index)):
    CT = CTD_method.iloc[j]['CT method']
    TD = CTD_method.iloc[j]['TD database']
    
    if graph.has_edge(CT, TD):
        graph[CT][TD]['weight'] += 1
    
    else:
        graph.add_edge(CT,TD,weight =1)
        
HCTD_network = nx.to_pandas_edgelist(graph)

HCTD_network.to_excel('HCTD network edgelist.xlsx')

# node attribute

HC = pd.DataFrame(np.array(df['HC method']), columns = ['Method']).dropna(axis=0)
HC['Frequency'] = 1
HC_pivot = HC.pivot_table(index = 'Method', values='Frequency', aggfunc='sum')
HC_pivot= HC_pivot.sort_values('Frequency', ascending=False)
HC_pivot['type'] = 'HC'

CT = pd.DataFrame(np.array(df['CT method']), columns = ['Method']).dropna(axis=0)
CT['Frequency'] = 1
CT_pivot = CT.pivot_table(index = 'Method', values='Frequency', aggfunc='sum')
CT_pivot= CT_pivot.sort_values('Frequency', ascending=False)
CT_pivot['type'] = 'CT'

TD = pd.DataFrame(list(df['TD database']), columns = ['Method']).dropna(axis=0)
TD['Frequency'] = 1
TD_pivot = TD.pivot_table(index = 'Method', values='Frequency', aggfunc='sum')
TD_pivot= TD_pivot.sort_values('Frequency', ascending=False)
TD_pivot['type'] = 'TD'

HCTD_att = pd.concat([HC_pivot, CT_pivot, TD_pivot])

HCTD_att.to_excel('HCTD network attr.xlsx')


    