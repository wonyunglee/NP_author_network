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
import holoviews as hv
from holoviews import opts, dim
import matplotlib as mpl

sns.set(style="white", context="talk")

df = pd.read_excel('190228_netpharm_list (2018).xlsx', sheet_name='drug_availavility,DTIs_methods', index_col='약물')

threshold = 5

#### affiliation, H-C, C-T, T-D database, method frequency

def Frequency(df_name, column_name, split_by):
    list_name = []
    
    df_name = df_name[[column_name,'year']].dropna()
    
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
    
    ax = fre_year.plot(kind='bar', stacked=True, figsize= (8,6), fontsize = 15, legend= 'reverse', width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), prop={'size':15}) 
    ax.set_yticks([i*y_ticks for i in range(max_value//y_ticks +2)])
    
    plt.xlabel('')
    plt.ylim(0,((max_value//10)*10 +11))
        
    plt.savefig(str(save_fig_name))
    plt.show()
    plt.close()
    
    return ax

# Frequency(df_name, column_name, split_by):
# stacked_barplot_visual(pivot_table, column, threshold, y_ticks, save_fig_name)


aff = Frequency(df, 'affiliation', ",")
aff_bar = stacked_barplot_visual(aff,'affiliation',5,5,'0.affiliation fre.png' )


HC = Frequency(df, 'HC method', ", ")
HC_bar = stacked_barplot_visual(HC, 'HC method', 5,10,'1. HC fre.png')

CT = Frequency(df, 'CT method', ', ')
CT_bar = stacked_barplot_visual(CT, 'CT method', 5, 10,'2. CT fre.png')

TD = Frequency(df, 'TD database', ', ')
TD_bar = stacked_barplot_visual(TD, 'TD database', 5, 10,'3. TD fre.png')













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
HC_threshold = (HC_pivot_fre['Frequency'] >= threshold)
HC_pivot_fre = HC_pivot_fre.loc[HC_threshold]

HC_pivot_fre_year = HC_fre.pivot_table(index=['HC_method','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
HC_pivot_fre_year = pd.DataFrame(HC_pivot_fre_year.to_records())

HC_pivot_fre_year = HC_pivot_fre_year.loc[HC_pivot_fre_year['HC_method'].isin(list(HC_pivot_fre.index))]


# make pivot table HC by frequency, year 

HC_fre_year = HC_pivot_fre_year.pivot_table(index=['HC_method'],columns=['year'], aggfunc=np.sum, fill_value=0)
years = [i for i in range (HC_fre_year.columns[0][1],HC_fre_year.columns[-1][1]+1)]
HC_fre_year.columns = years
HC_fre_year['sum'] = HC_fre_year.sum(axis=1)

HC_threshold = (HC_fre_year['sum'] >= threshold)
HC_fre_year = HC_fre_year.loc[HC_threshold]

HC_fre_year = HC_fre_year.sort_values(by = 'sum', ascending=False)
del HC_fre_year['sum']

# visualization HC frequency by stacked bar plot

ax = HC_fre_year.plot(kind='bar', stacked=True, figsize= (12,10), fontsize = 20, legend= 'reverse', width=0.8)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='upper right',prop={'size':20}) 

plt.xlabel('')

#plt.savefig('1. HC by fre.png')
plt.show()
plt.close()

# Drug availability assessment

DA_year = df[['year','Drug availability','OB & DL']]

DA_year_pivot = DA_year.pivot_table(index='year')


fig, ax = plt.subplots()
fig.set_size_inches(12,10)
ax.set_ylabel('Proportion')
sns.lineplot(data=DA_year_pivot)


#plt.savefig('1. DA ratio.png')
plt.show()
plt.close()




#
#
## HC by frequency
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#
#ax = sns.barplot(x=HC_pivot_fre.index, y=HC_pivot_fre['Frequency'])
#plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
#plt.savefig('1. HC by fre.png')
#plt.show()
#plt.close()
#
## HC by year
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#ax = sns.lineplot(data=HC_pivot_fre_year, x='year', y='Frequency', hue='HC_method')
#
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[1:], labels=labels[1:])
#
#plt.setp(ax.get_legend().get_texts(), fontsize='15')
#plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
#plt.savefig('1. HC by year.png')
#plt.show()
#plt.close()

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
CT_threshold = (CT_pivot_fre['Frequency'] >= threshold)
CT_pivot_fre = CT_pivot_fre.loc[CT_threshold]

CT_pivot_fre_year = CT_fre.pivot_table(index=['CT_method','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
CT_pivot_fre_year = pd.DataFrame(CT_pivot_fre_year.to_records())

CT_pivot_fre_year = CT_pivot_fre_year.loc[CT_pivot_fre_year['CT_method'].isin(list(CT_pivot_fre.index))]


# make pivot table CT by frequency, year 

CT_fre_year = CT_pivot_fre_year.pivot_table(index=['CT_method'],columns=['year'], aggfunc=np.sum, fill_value=0)
years = [i for i in range (CT_fre_year.columns[0][1],CT_fre_year.columns[-1][1]+1)]
CT_fre_year.columns = years
CT_fre_year['sum'] = CT_fre_year.sum(axis=1)

CT_threshold = (CT_fre_year['sum'] >= threshold)
CT_fre_year = CT_fre_year.loc[CT_threshold]

CT_fre_year = CT_fre_year.sort_values(by = 'sum', ascending=False)
del CT_fre_year['sum']

# visualization CT frequency by stacked bar plot

ax = CT_fre_year.plot(kind='bar', stacked=True, figsize= (12,10), fontsize = 20, legend= 'reverse', width=0.8)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='upper right',prop={'size':20}) 

plt.xlabel('')

#plt.savefig('2. CT by fre.png')
plt.show()
plt.close()



#
#
## CT by fre
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#
#ax = sns.barplot(x=CT_pivot_fre.index, y=CT_pivot_fre['Frequency'])
#plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
##plt.savefig('2. CT by fre.png') 이건 별로!
#plt.show()
#plt.close()
#
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#ax = sns.lineplot(data=CT_pivot_fre_year, x='year', y='Frequency', hue='CT_method')
#
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[1:], labels=labels[1:])
#
#plt.setp(ax.get_legend().get_texts(), fontsize='15')
#plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
#plt.savefig('2. CT by year.png')
#
#plt.show()
#
#plt.close()

### CT method- subgroup visualization using donut plot

CT_pivot_fre = CT_fre.pivot_table(index='CT_method',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)

CT_pivot_fre['name'] = CT_pivot_fre.index

df_CT = pd.read_excel('190228_netpharm_list (2018).xlsx', sheet_name='Compound-target interaction')

CT_pivot_fre = CT_pivot_fre.merge(df_CT[['name','type']])
CT_pivot_fre = CT_pivot_fre.sort_values(by= ['Frequency'],ascending=False)

CT_type_names = ['Chemogenomic approach', 'Docking simulation approach','Ligand-based approach', 'Others']

CT_type_pivot = CT_pivot_fre.pivot_table(index='type',values='Frequency',aggfunc=np.sum)
CT_type_size = list(CT_type_pivot['Frequency'])


# Create colors 
a, b, c, d =  [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.YlOrBr]
color_list = [a,b,c,d]


# subgroup name, size, color
subgroup_names = []
subgroup_size = []
subgroup_color = []

method_threshold = 3   # threshold 이상인 method만 보이게 / 1은 합쳐서 others 처

for i in range(len(CT_type_names)):

    type_is = (CT_pivot_fre['type'] == CT_type_names[i])
    df_subgroup_raw = CT_pivot_fre.loc[type_is]
    
    df_over_threshold = df_subgroup_raw['Frequency'] >= method_threshold
    df_subgroup = df_subgroup_raw.loc[df_over_threshold]
    
    subgroup_names.extend(list(df_subgroup['name']))
    subgroup_size.extend(list(df_subgroup['Frequency']))

    for j in range(len(df_subgroup.index)):
        subgroup_color.append(color_list[i](0.7-0.07*j))
    
    if len(df_over_threshold) != df_over_threshold.sum():  # if isin false in dataframe
        subgroup_names.append('Others')
        subgroup_size.append(df_subgroup_raw['Frequency'].loc[~df_over_threshold].sum())
        subgroup_color.append(color_list[i](0.1))



# First ring

fig, ax = plt.subplots() 
fig.set_size_inches(8,8)

ax.axis('equal') 
mypie, _ = ax.pie(CT_type_size, radius=1.3, labels=['','','',''], colors=[a(0.7), b(0.7), c(0.7),d(0.7)], startangle = -280) 
# mypie, _ = ax.pie(CT_type_size, radius=1.3-0.3, labels=['','','',''], colors=[a(0.7), b(0.7), c(0.7),d(0.7)], startangle = -275) 

plt.setp( mypie, width=1.3, edgecolor='white') 

# Second ring

mypie2, texts = ax.pie(subgroup_size, radius=1.3, labels=subgroup_names, labeldistance=1.05, colors =subgroup_color, startangle = -280) 

plt.setp( mypie2, width=0.3, edgecolor='white') 
plt.margins(0,0) 

plt.show()
#plt.savefig('2. CT fre by type.png')
plt.close()




## C-T category ratio by year

df_CT = pd.read_excel('190228_netpharm_list (2018).xlsx', sheet_name='Compound-target interaction')
df_CT = df_CT[['name','frequency','type']]
df_CT = df_CT.loc[df_CT['frequency']>=threshold]

# hue color
red = np.array([229,36,38]) / 256
green = np.array([62,150,81])/256
blue = np.array([32,114,178])/256
yellow = np.array([218,124,48])/256

palette = {'Chemogenomic approach': red, 'Docking simulation approach':green,'Ligand-based approach' :blue, 'Others':yellow}

fig, ax = plt.subplots()
fig.set_size_inches(12,10)

g = sns.barplot(x= 'name' , y= 'frequency' , hue='type' , data = df_CT, dodge=False, palette=palette)

leg = ax.legend()
leg.set_title('')

plt.setp(ax.get_legend().get_texts(), fontsize='20')
plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
plt.setp(ax.get_yticklabels(), fontsize='20')

ax.set_xlabel('')
ax.set_ylabel('')


plt.show()

#plt.savefig('2. CT by fre.png')
plt.close()




### CT co-occurrence chord diagram


# CT method co-occurrence network

CT_method = df['CT method']
CT_method.dropna(axis=0, inplace=True)

G_CT = nx.Graph()

for i in range (len(CT_method.index)):
        
    CT = CT_method.iloc[i].split(", ")
    
    if len(CT) == 1:
        pass
    else:
        for j,k in itertools.combinations(CT,2):
            
            if G_CT.has_edge(j,k):
                G_CT[j][k]['weight'] += 1

            else:
                G_CT.add_edge(j,k,weight = 1 )

CT_co_occu = nx.to_pandas_edgelist(G_CT)                

CT_co_occu.to_excel('4. CT co_occurrence.xlsx')    




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
TD_threshold = (TD_pivot_fre['Frequency'] >= threshold)
TD_pivot_fre = TD_pivot_fre.loc[TD_threshold]

TD_pivot_fre_year = TD_fre.pivot_table(index=['TD_database','year'],aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
TD_pivot_fre_year = pd.DataFrame(TD_pivot_fre_year.to_records())

TD_pivot_fre_year = TD_pivot_fre_year.loc[TD_pivot_fre_year['TD_database'].isin(list(TD_pivot_fre.index))]



# make pivot table TD by frequency, year 

TD_fre_year = TD_pivot_fre_year.pivot_table(index=['TD_database'],columns=['year'], aggfunc=np.sum, fill_value=0)
years = [i for i in range (TD_fre_year.columns[0][1],TD_fre_year.columns[-1][1]+1)]
TD_fre_year.columns = years
TD_fre_year['sum'] = TD_fre_year.sum(axis=1)

TD_threshold = (TD_fre_year['sum'] >= threshold)
TD_fre_year = TD_fre_year.loc[TD_threshold]

TD_fre_year = TD_fre_year.sort_values(by = 'sum', ascending=False)
del TD_fre_year['sum']

# visualization TD frequency by stacked bar plot

ax = TD_fre_year.plot(kind='bar', stacked=True, figsize= (12,10), fontsize = 20, legend= 'reverse', width=0.8)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='upper right',prop={'size':20}) 

plt.xlabel('')

#plt.savefig('3. TD by fre.png')
plt.show()
plt.close()



TD_pivot_fre = TD_fre.pivot_table(index='TD_database',aggfunc='sum', values = 'Frequency').sort_values(by = 'Frequency',ascending=False)
#TD_threshold = (TD_pivot_fre['Frequency'] >= threshold)
#TD_pivot_fre = TD_pivot_fre.loc[TD_threshold]


### TD method- subgroup visualization using donut plot

TD_pivot_fre['name'] = TD_pivot_fre.index

df_TD = pd.read_excel('190228_netpharm_list (2018).xlsx', sheet_name='target-disease interaction')

TD_pivot_fre = TD_pivot_fre.merge(df_TD[['name','type']])
TD_pivot_fre = TD_pivot_fre.sort_values(by= ['Frequency'],ascending=False)

TD_type_names = ['Biological function', 'Pathway','Disease']

TD_type_pivot = TD_pivot_fre.pivot_table(index='type',values='Frequency',aggfunc=np.sum)
TD_type_size = list(TD_type_pivot['Frequency'].loc[TD_type_names[i]] for i in range(len(TD_type_names)))

# Create colors 
a, b, c=  [plt.cm.Purples, plt.cm.Oranges, plt.cm.GnBu]
color_list = [a,b,c]


# subgroup name, size, color
subgroup_names = []
subgroup_size = []
subgroup_color = []

method_threshold = 3   # 2 이상인 method만 보이게 / 1은 합쳐서 others 

for i in range(len(TD_type_names)):
ram
    type_is = (TD_pivot_fre['type'] == TD_type_names[i])
    df_subgroup_raw = TD_pivot_fre.loc[type_is]
    
    df_over_threshold = df_subgroup_raw['Frequency'] >= method_threshold
    df_subgroup = df_subgroup_raw.loc[df_over_threshold]
    
    subgroup_names.extend(list(df_subgroup['name']))
    subgroup_size.extend(list(df_subgroup['Frequency']))

    for j in range(len(df_subgroup.index)):
        subgroup_color.append(color_list[i](0.7-0.1*j))
    
    if len(df_over_threshold) != df_over_threshold.sum():  # if isin false in datafe
        subgroup_names.append('Others')
        subgroup_size.append(df_subgroup_raw['Frequency'].loc[~df_over_threshold].sum())
        subgroup_color.append(color_list[i](0))

# First ring

fig, ax = plt.subplots() 
fig.set_size_inches(8,8)

ax.axis('equal') 
mypie, _ = ax.pie(TD_type_size, radius=1.3, labels=['','',''], colors=[a(0.7), b(0.7), c(0.7)], startangle = 0) 
plt.setp( mypie, width=1.3, edgecolor='white') 

#mypie, _ = ax.pie(TD_type_size, radius=1.3-0.3, labels=['','',''], colors=[a(0.7), b(0.7), c(0.7)], startangle = 0) 
#plt.setp( mypie, width=0.3, edgecolor='white') 


# Second ring

mypie2, texts = ax.pie(subgroup_size, radius=1.3, labels=subgroup_names, labeldistance=1.1, colors =subgroup_color, startangle = 0) 

plt.setp( mypie2, width=0.3, edgecolor='white')  # second ring width 조
plt.margins(0,0) 

plt.show()

#plt.savefig('3. TD fre by type.png')
plt.close()






#
#
## TD frequency
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#
#ax = sns.barplot(x=TD_pivot_fre.index, y=TD_pivot_fre['Frequency'])
#plt.setp(ax.get_xticklabels(), rotation=90, fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
#plt.savefig('3. TD by fre.png')
#plt.show()
#plt.close()
#
## TD frequency by year
#
#fig, ax = plt.subplots()
#fig.set_size_inches(12,10)
#
#g = sns.lineplot(data=TD_pivot_fre_year, x='year', y='Frequency', hue='TD_database')
#
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[1:], labels=labels[1:])
#
#plt.setp(ax.get_legend().get_texts(), fontsize='20')
#plt.setp(ax.get_xticklabels(), fontsize='20')
#plt.setp(ax.get_yticklabels(), fontsize='20')
#
#
#ax.set_xlabel('')
#ax.set_ylabel('')
#
#plt.savefig('3. TD by year.png')
#plt.show()
#plt.close()


### CT method- subgroup visualization using donut plot







### H-C-T-D network construction


# node attribute

threshold = (HC_pivot_fre['Frequency'] > 5)
HC_pivot= HC_pivot_fre.loc[threshold]
HC_pivot['type'] = 'HC'
HC_pivot['name'] = HC_pivot.index

threshold = (CT_pivot_fre['Frequency'] > 5)
CT_pivot= CT_pivot_fre.loc[threshold]
CT_pivot['type'] = 'CT'

threshold = (TD_pivot_fre['Frequency'] > 5)
TD_pivot = TD_pivot_fre.loc[threshold] 
TD_pivot['type'] = 'TD'

HCTD_att = pd.concat([HC_pivot, CT_pivot, TD_pivot])
HCTD_att = HCTD_att.reset_index(drop=True)

HCTD_att.to_excel('5. HCTD network attr.xlsx')


# Graph construction

graph = nx.Graph()

# CT_pair

HCT_method = df[['HC method','CT method']]
HCT_method.dropna(axis=0, inplace=True)

for i in range(len(HCT_method.index)):
    HC = HCT_method.iloc[i]['HC method'].split(", ")
    CT = HCT_method.iloc[i]['CT method'].split(", ")

    for j in range(len(HC)):
        
        HC_pair = HC[j]
        
        for p in range(len(CT)):
            
            CT_pair = CT[p]
            
            if graph.has_edge(HC_pair, CT_pair):
                
                graph[HC_pair][CT_pair]['weight'] += 1
        
            else:
                graph.add_edge(HC_pair,CT_pair, weight=1)

    
CTD_method = df[['CT method','TD database']]
CTD_method.dropna(axis=0, inplace=True)


for i in range(len(CTD_method.index)):
    CT = CTD_method.iloc[i]['CT method'].split(", ")
    TD = CTD_method.iloc[i]['TD database'].split(", ")
    
    for j in range(len(CT)):
        
        CT_pair = CT[j]
        
        for p in range(len(TD)):
            
            TD_pair = TD[p]
            
            if graph.has_edge(CT_pair, TD_pair):
                
                graph[CT_pair][TD_pair]['weight'] += 1
            
            else:
                graph.add_edge(CT_pair, TD_pair, weight=1)

# remove nodes with frequency 1

nodes = set(graph.nodes())
threshold_node = set(HCTD_att['name'])
graph.remove_nodes_from(list(nodes-threshold_node))

HCTD_network = nx.to_pandas_edgelist(graph)

HCTD_network.to_excel('5. HCTD network edgelist.xlsx')





    