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

def Frequency(df_name, column_name, split_by):
    list_name = []
    
    for i in range(len(df_name.index)):
        
        name = df_name[column_name].iloc[i].split(split_by)
        
        if type(name) == str:
            name = [name]
        
        list_name.extend(name)
            
    list_name = list(set(list_name))    
    
    df_year_fre = pd.DataFrame(np.zeros((len(list_name),3)), columns=[column_name,'year','Frequency'])
    
    k=0
    
    for j in range(len(df_name.index)):
        
        element = df_name[column_name].iloc[j].split(split_by)
        year = df_name['year'].iloc[j]
        
        if type(name) == str:
            element = [element]
        
        for p in range(len(element)):
         
            df_year_fre.loc[k] = [element[p],year,1]
            k+=1
            
    pivot_fre_year = df_year_fre.pivot_table(index = [column_name,'year'], aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
    pivot_fre_year = pd.DataFrame(pivot_fre_year.to_records())
        
    return pivot_fre_year

def frequency_over_threshold(pivot_table, column, threshold):
        
    pivot = pivot_table.pivot_table(index='affiliation',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency', ascending = False)
    over_threshold = pivot.index[(pivot['Frequency'] >= threshold)]
    over_threshold_row = pivot.index.isin(list(over_threshold))
    pivot = pivot_table.loc[over_threshold_row]
    pivot = pivot.reset_index(drop=True)
    
    return pivot

def stacked_barplot_visual(pivot_table, column, threshold, y_ticks, save_fig_name):
    
    fre_year = pivot_table.pivot_table(index=[str(column)],columns=['year'], aggfunc=np.sum, fill_value=0)
    
    years = [i for i in range (int(fre_year.columns[0][1]),int(fre_year.columns[-1][1]+1))]
    fre_year.columns = years
    fre_year['sum'] = fre_year.sum(axis=1)

    over_threshold = (fre_year['sum'] >= threshold)
    fre_year = fre_year.loc[over_threshold]
    fre_year = fre_year.sort_values(by='sum', ascending = True)
    
    max_value = fre_year['sum'].iloc[-1]
    
    del fre_year['sum']
    
    ax = fre_year.plot(kind='bar', stacked=True, figsize= (12,10), fontsize = 20, legend= 'reverse', width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), prop={'size':20}) 
    ax.set_yticks([i*y_ticks for i in range(max_value//y_ticks +2)])
    
    plt.xlabel('')
    plt.ylim(0,((max_value//10)*10 +11))
        
    plt.savefig(str(save_fig_name))
    plt.show()
    plt.close()
    
    return ax

aff = Frequency(df, 'affiliation', ",")

aff_bar = stacked_barplot_visual(aff,'affiliation',5,5,'0.affiliation fre.png' )


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



        

co_author_list = list_generation(df, 'corresponding au', ",")











df = pd.read_excel('190228_netpharm_list (2018).xlsx', sheetname='drug_availavility,DTIs_methods')

df_author_aff = df[['author','affiliation','year','Drug availability','OB & DL', 'Chemogenomic approach', 'Docking simulation approach','Ligand-based approach','Others']]



    