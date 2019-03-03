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

df = pd.read_excel('190228_netpharm_list (2018).xlsx', sheet_name='drug_availavility,DTIs_methods', index_col='약물')







### co-authorship network construction 

# author network edge list 

G  = nx.Graph()

for i in range(len(df.index)):

    auth = df['author'].iloc[i].split(",")
    for j in itertools.combinations(auth,2):
        G.add_edge(*j)

        
author_net = nx.to_pandas_edgelist(G)
author_net['early'] = 0
author_net['late'] = 0
G = nx.from_pandas_edgelist(author_net, edge_attr=['early','late'])    

# update weight of author network by time 

early = (df['year']<=2014)
late = (df['year']>2014)

period = {0:early, 1:late}

for q in range(len(period)):
    
    df_period = df.loc[period[q]]
    
    if q == 0:
        time = 'early'
    else:
        time = 'late'
    
    for i in range(len(df_period.index)):
        
        auth = df_period['author'].iloc[i].split(",")
    
        for j in itertools.combinations(auth,2):
            
            G[j[0]][j[1]][time] += 1

author_net = nx.to_pandas_edgelist(G)
author_net['total'] = author_net['early'] + author_net['late']
author_net.to_excel('1. author_network.xlsx')

# author frequency

author_list = pd.DataFrame(index=list(G.nodes))

author_list['corr'] = 0

author_list['corr_fre'] = 0
author_list['corr_early'] = 0
author_list['corr_late'] = 0

author_list['fre'] = 0 
author_list['early'] = 0
author_list['late'] = 0

author_list.loc['Benjiao Gong'] = [0,0,0,0,0,0,0]  # 단독 저자
author_list.loc['Xianxiang Liu'] = [0,0,0,0,0,0,0]  # 단독 저자

early = (df['year']<=2014)
late = (df['year']>2014)

period = {0:early, 1:late}

for q in range(len(period)):
    
    df_period = df.loc[period[q]]
    
    if q ==0:
        time = 'corr_early'
    else:
        time = 'corr_late'

    for i in range(len(df_period.index)):
        corr = df_period['corresponding au'].iloc[i].split(",")
        
        if type(corr) == str:
            author_list['corr'].loc[corr] = 1
            author_list[time].loc[corr] += 1
            
        else:
            for j in range(len(corr)):
                
                author_list['corr'].loc[corr[j]] = 1
                author_list[time].loc[corr[j]] += 1            
        
author_list['corr_fre'] = author_list['corr_early'] + author_list['corr_late']


for q in range(len(period)):
    
    df_period = df.loc[period[q]]
    
    if q ==0:
        time = 'early'
    else:
        time = 'late'

    for i in range(len(df_period.index)):
        corr = df_period['author'].iloc[i].split(",")
        
        if type(corr) == str:
            author_list[time].loc[corr] += 1
            
        else:
            for j in range(len(corr)):
                
                author_list[time].loc[corr[j]] += 1            
        
author_list['fre'] = author_list['early'] + author_list['late']


author_list.to_excel('1. corr_frequency.xlsx')
 

### Affliation network construction

G_aff  = nx.Graph()

for i in range(len(df.index)):

    aff = df['affiliation'].iloc[i].split(",")
    if type(aff) == str:
        pass

    for j in itertools.combinations(aff,2):
        G_aff.add_edge(*j)
       
aff_net = nx.to_pandas_edgelist(G_aff)

aff_net['early'] = 0
aff_net['late'] = 0

G_aff = nx.from_pandas_edgelist(aff_net, edge_attr=['early','late'])    

# aff network time attr

early = (df['year']<=2014)
late = (df['year']>2014)

period = {0:early, 1:late}

for q in range(len(period)):
    
    df_period = df.loc[period[q]]
    
    if q == 0:
        time = 'early'
    else:
        time = 'late'
    
    for i in range(len(df_period.index)):
        
        aff = df_period['affiliation'].iloc[i].split(",")
        
        if type(aff) == str:
            pass
        
        for j in itertools.combinations(aff,2):
            
            G_aff[j[0]][j[1]][time] += 1

aff_net = nx.to_pandas_edgelist(G_aff)
aff_net['total'] = aff_net['early'] + aff_net['late']
aff_net.to_excel('2. aff_network.xlsx')

# affiliation frequency

aff_list = []

for i in range(len(df.index)): 
    aff = df['affiliation'].iloc[i].split(",")
    
    if type(aff) == str:
        aff_list.append(aff)
        
    else:
        for j in range(len(aff)):
            aff_list.append(aff[j])

aff_list = pd.DataFrame(index=list(set(aff_list)))

aff_list['early'] = 0
aff_list['late'] = 0

early = (df['year']<=2014)
late = (df['year']>2014)

period = {0:early, 1:late}

for q in range(len(period)):
    
    df_period = df.loc[period[q]]
    
    if q ==0:
        time = 'early'
    else:
        time = 'late'

    for i in range(len(df_period.index)):
        aff = df_period['affiliation'].iloc[i].split(",")
        
        if type(aff) == str:
            aff = [aff]

        for j in range(len(aff)):
            aff_list[time].loc[aff[j]] += 1            
                
aff_list['total'] = aff_list['early'] + aff_list['late']
aff_list.to_excel('2. aff_frequency.xlsx')





#### affiliation, paper by year

aff_paper_year = pd.DataFrame(np.array(np.zeros((8,2))),columns = (['year','affiliation']), 
                              index = ([i for i in range(2011,2019)]))

aff_df = pd.DataFrame(df[['affiliation','year']])

for i in range(2012,2019):
    
    aff_group = (aff_df['year'] == i)
    aff_group = aff_df[aff_group]
    
    affiliation_list = []
    
    for j in range(len(aff_group.index)):
        
        aff = df_period['affiliation'].iloc[j].split(",")
        
        for p in range(len(aff)):
            
            affiliation_list.append(aff[p])
        
    affiliation_list = list(set(affiliation_list))
    aff_paper_year['affiliation'].loc[i] = len(affiliation_list)

aff_paper_year['affiliation'].loc[2011] = 3  # 2011년은 manually 추가

paper =  pd.DataFrame(df['year'])
paper['1'] = 1
pivot_3 = paper.pivot_table(index='year',values='1', aggfunc='sum')
pivot_3.loc[2011] = 2 

aff_paper_year['year'] = pivot_3['1']

# visualization


red = tuple(np.array([229,36,38]) / 256)
blue = tuple(np.array([32,114,178])/256)

plt.plot(aff_paper_year['year'], color =blue )
plt.plot(aff_paper_year['affiliation'], color=red)
plt.legend(['Affiliation', 'Paper'])
plt.show()


### drug-availability(DA) and drug-target interaction (DTI) ratio visualization

df_method = pd.read_excel('190228_netpharm_list (2018).xlsx', sheetname='drug_availavility,DTIs_methods')

# DA visualization

df_DA = df_method[['year','Drug availability','OB & DL']]
df_DA_ratio = pd.DataFrame(np.array(np.zeros((7,2))),columns = (['Drug availability','OB & DL']), 
                              index = ([i for i in range(2012,2019)]))

for i in range(2012,2019):
    
    DA_method = (df_DA['year'] == i)
    DA_method = df_DA[DA_method]
    
    ratio_DA = DA_method['Drug availability'].sum() / len(DA_method.index)
    ratio_OBDL = DA_method['OB & DL'].sum() / len(DA_method.index)
    
    df_DA_ratio.loc[i] = np.array([[ratio_DA,ratio_OBDL]])
    

fig, ax = plt.subplots()
fig.set_size_inches(12,10)

plt.plot(df_DA_ratio['Drug availability'], 'coral')
plt.plot(df_DA_ratio['OB & DL'], 'r')

plt.setp(ax.get_xticklabels(), fontsize='20')
plt.setp(ax.get_yticklabels(), fontsize='20')

plt.legend(['Drug availability', 'OB & DL'])
plt.savefig('4. DA_ratio.png')
plt.show()
plt.close()

df_DA_ratio.to_excel('4. DA_ratio.xlsx')


# DTI visualization

df_DTI_ratio = pd.DataFrame(np.array(np.zeros((7,4))),columns = (['Chemogenomic approach','Docking simulation approach','Ligand-based approach', 'Experimental approach']), 
                              index = ([i for i in range(2012,2019)]))

df_DTI = pd.read_excel('190228_netpharm_list (2018).xlsx', sheetname='drug_availavility,DTIs_methods')

df_DTI = df_DTI[['year','Chemogenomic approach','Docking simulation approach','Ligand-based approach', 'Experimental approach']]


for j in range(2012,2019):
    
    DTI_method = (df_DTI['year'] == j)
    DTI_method = df_DTI[DTI_method]
    
    DTI_sum = DTI_method[['Chemogenomic approach','Docking simulation approach','Ligand-based approach', 'Experimental approach']]
    DTI_sum = DTI_sum.sum().sum()
    
    ratio_chemogenomic = DTI_method['Chemogenomic approach'].sum() / DTI_sum
    ratio_docking = DTI_method['Docking simulation approach'].sum() / DTI_sum
    ratio_ligand = DTI_method['Ligand-based approach'].sum() / DTI_sum
    ratio_experiment = DTI_method['Experimental approach'].sum() / DTI_sum


    df_DTI_ratio.loc[j] = np.array([[ratio_chemogenomic,ratio_docking,ratio_ligand, ratio_experiment]])


red = np.array([229,36,38]) / 256
green = np.array([62,150,81])/256
blue = np.array([32,114,178])/256
yellow = np.array([218,124,48])/256

fig, ax = plt.subplots()
fig.set_size_inches(12,10)

plt.plot(df_DTI_ratio['Chemogenomic approach'], color= red)
plt.plot(df_DTI_ratio['Docking simulation approach'], color= green)
plt.plot(df_DTI_ratio['Ligand-based approach'], color= blue)
plt.plot(df_DTI_ratio['Experimental approach'], color= yellow)


plt.setp(ax.get_xticklabels(), fontsize='20')
plt.setp(ax.get_yticklabels(), fontsize='20')

plt.legend(['Chemogenomic approach','Docking simulation approach','Ligand-based approach', 'Experimental approach'])
plt.show('4. DTI_ratio.png')
    
df_DTI_ratio.to_excel('4. DTI_ratio.xlsx')
 

#### DA, DTIs ratio attributes by author, affiliation 

## DA

df = pd.read_excel('190228_netpharm_list (2018).xlsx', sheetname='drug_availavility,DTIs_methods')
df_author = df[['corresponding au','affiliation','year','Drug availability','OB & DL', 'Chemogenomic approach', 'Docking simulation approach','Ligand-based approach','Others']]
author_list = []

# author list generation

df_author_DA = pd.DataFrame(columns = ['corresponding au','total','Drug availability','OB & DL'])







###### DA, DTIs ratio by index

def DA_DTI_ratio(df, index_column, year_from, year_to):
    # DA (OB, DL)은 사용 여부 / DTIs type은 빈도를 보내줌 
    
    df = df.reset_index(drop=True)
    year_range =  df['year'].isin(i for i in range( year_from, year_to+1))
    df = df.loc[year_range]
    
    index_list = []
    
    for i in range(len(df.index)):
        
        index = df[index_column].iloc[i].split(",")
        
        if type(index) == str:
            index = [index]
            
        for j in range(len(index)):
            
            index_list.extend(index)
            
    index_list = list(set(index_list))
    
    DA_method = pd.DataFrame(np.zeros((len(index_list),6)), index = index_list, columns = ['Drug availability','OB & DL','Chemogenomic approach','Docking simulation approach','Ligand-based approach','Others'])
    
    for p in range(len(df.index)):
        
        index = df[index_column].iloc[p].split(",")
        
        method = df[list(DA_method.columns)].iloc[p]
        
        if type(index) == str:
            index = [index]
        
        for q in range(len(index)):
            
            DA_method.loc[index[q]] += method
    
    
    DA_array = np.array(DA_method[['Drug availability','OB & DL']])
    over_one = DA_array >= 1
 
    DA_array[over_one] = 1
    DA_type = DA_array.sum(axis=1)
    
    DA_method[['Drug availability','OB & DL']] = DA_array
    DA_method['DA type'] =DA_type
    
    return DA_method
    
# DA, DTI ratio of co-author

co_early = DA_DTI_ratio(df, 'corresponding au', 2011,2014)
co_early.to_excel('0. DA,DTI_co_au_early.xlsx')

co_late = DA_DTI_ratio(df, 'corresponding au', 2015,2018)
co_late.to_excel('0. DA,DTI_co_au_late.xlsx')

co_total = DA_DTI_ratio(df, 'corresponding au', 2011,2018)
co_total.to_excel('0. DA,DTI_co_au_total.xlsx')

aff_early = DA_DTI_ratio(df, 'affiliation', 2011,2014)
aff_early.to_excel('0. DA,DTI_aff_early.xlsx')

aff_late= DA_DTI_ratio(df, 'affiliation', 2015,2018)
aff_late.to_excel('0. DA,DTI_aff_late.xlsx')

aff_total = DA_DTI_ratio(df, 'affiliation', 2011,2018)
aff_total.to_excel('0. DA,DTI_aff_total.xlsx')






