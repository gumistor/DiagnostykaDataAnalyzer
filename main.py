import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import matplotlib.patches as mpatches
import os

sns.set(style='darkgrid')



fileList = os.listdir(r'data')

df = None

for fileItem in fileList:
    fileName = os.path.join('.\\data', fileItem)

    if df is None:
        df = pd.read_csv(fileName, sep=';', decimal=",")
    else:
        df1 = pd.read_csv(fileName, sep=';', decimal=",")
        df = pd.concat([df, df1], axis=0)

# drop duplicates
df = df.drop_duplicates()
df.reset_index(inplace=True, drop=True)

unique_parameters = df['Parametr'].unique()

for unique_parameter_item in unique_parameters:
    df1 = df[df['Parametr'] == unique_parameter_item]
    if df1['Badanie'].unique().shape > (1,):
        # print("Sprawdzić czy nie są to różne badania: ")
        # print(df1['Badanie'].unique())
        pass

# drop column with 'Badanie' i 'Kod zlecenia'
# df = df.drop(labels=['Badanie', 'Kod zlecenia','Zakres referencyjny'], axis=1)

# df[['Data', 'Czas']] = df['Data'].str.split(' ', 1, expand=True)
df[['Wynik', 'Jednostka']] = df.loc[:, 'Wynik'].str.split(' ', 1, expand=True)
df[['Min', 'Max']] = df.loc[:, 'Zakres referencyjny'].str.split(' - ', 1, expand=True)

df = df.drop(labels=['Badanie', 'Kod zlecenia', 'Zakres referencyjny', 'Opis'], axis=1)

print(df.columns)

df.loc[:, 'Data'] = pd.to_datetime(df.loc[:, 'Data'], format='%d-%m-%Y %H:%M:%S')

df.loc[df['Min'] == '', 'Min'] = "0,0"
df.loc[df['Max'] == '', 'Max'] = str(np.inf)

df.loc[:, 'Wynik'] = df.loc[:, 'Wynik'].str.replace(",", ".").astype(float)
df.loc[:, 'Min'] = df.loc[:, 'Min'].str.replace(",", ".").astype(float)
df.loc[:, 'Max'] = df.loc[:, 'Max'].str.replace(",", ".").astype(float)

df_ch = df.loc[df['Parametr'] == 'Cholesterol całkowity', ('Data','Wynik')]


def calc(item):
    global df

    chol_cal = (df.loc[:, 'Data'] == item.Data) & (df.loc[:, 'Parametr'] == 'Cholesterol całkowity')
    chol_hdl = (df.loc[:, 'Data'] == item.Data) & (df.loc[:, 'Parametr'] == 'Cholesterol HDL')
    chol_ldl = (df.loc[:, 'Data'] == item.Data) & (df.loc[:, 'Parametr'] == 'Cholesterol LDL')
    chol_tru = (df.loc[:, 'Data'] == item.Data) & (df.loc[:, 'Parametr'] == 'Trójglicerydy')

    result = df.loc[chol_cal, 'Wynik'].values[0] / df.loc[chol_hdl, 'Wynik'].values[0]
    df_new1 = pd.DataFrame({'Parametr': 'Całkowity do HDL',
                           'Data':item.Data,
                           'Wynik': result,
                           'Jednostka': r'$\frac{mmol}{mmol}$',
                           'Min': 0,
                           'Max': np.inf}, index=[0])

    result = df.loc[chol_ldl, 'Wynik'].values[0] / df.loc[chol_hdl, 'Wynik'].values[0]

    df_new2 = pd.DataFrame({'Parametr': 'LDL do HDL',
                           'Data':item.Data,
                           'Wynik': result,
                           'Jednostka': r'$\frac{mmol}{mmol}$',
                           'Min': 0,
                           'Max': np.inf}, index=[0])

    result = 88.57 / 38.67 * df.loc[chol_tru, 'Wynik'].values[0] / df.loc[chol_hdl, 'Wynik'].values[0]
    df_new3 = pd.DataFrame({'Parametr': 'Trójglicerydy do HDL',
                           'Data':item.Data,
                           'Wynik': result,
                           'Jednostka': r'$\frac{mmol}{mmol}$',
                           'Min': 0,
                           'Max': np.inf}, index=[0])
    df = df.append(df_new1, ignore_index=True)
    df = df.append(df_new2, ignore_index=True)
    df = df.append(df_new3, ignore_index=True)


df_ch.apply(calc, axis=1)


def plot_parameter(df_local, param_name, label, ax_local, marker_in):
    alt = df.Parametr == param_name

    curr_data_local = df_local[alt].sort_values(axis=0, by=['Data'])

    curr_data_local.plot(x='Data', y='Wynik', xlabel='Data',
                         title=param_name, ylabel=curr_data_local.iloc[0]['Jednostka'],
                         ax=ax_local, grid=True, label=label, marker=marker_in, linestyle='dotted')

    return curr_data_local.loc[:, 'Data']

def plot_parameter_sea(df_local, param_name, ax_local):
    global sns

    alt = df.Parametr == param_name
    curr_data_local = df_local[alt].sort_values(axis=0, by=['Data'])

    sns.lineplot(x='Data', y='Wynik', data=curr_data_local, ax=ax_local)
    # , xlabel='Data' ,
    # title=param_name, ylabel=curr_data_local.iloc[0]['Jednostka'],
    # , grid=True, label=param_name, marker='.', linestyle='dotted')

    return curr_data_local.loc[:, 'Data']

def plot_set_of_values(title, values, labels, ax_local, last_plot=True, seaborn_enabled=False):

    markers = ['.', '>', '1', '<', '2', '^', '3', 'v', '4', '+', 'x', '|', '_']
    markers.reverse()

    if not labels:
        labels = values

    if seaborn_enabled:
        x_ticks_values = plot_parameter_sea(df, values[0], labels[0], ax_local)
    else:
        x_ticks_values = plot_parameter(df, values[0], labels[0], ax_local, markers.pop())

    for value, label in zip(values[1:], labels[1:]):
        x_ticks_values = pd.concat([x_ticks_values,
                                    plot_parameter(df, value,
                                                   label, ax_local, markers.pop())]).drop_duplicates().reset_index(
            drop=True).sort_values(axis=0)

    ax_local.set_title(title)
    ax_local.set_xticks(x_ticks_values)
    ax_local.set_xticklabels(x_ticks_values, rotation=45, fontsize=10)
    ax_local.legend(bbox_to_anchor=(1, 1), loc="upper left")

    date_form = DateFormatter("%y-%m-%d")
    ax_local.xaxis.set_major_formatter(date_form)

    if not last_plot:
        ax_local.set_xticklabels([])
        ax_local.set_xlabel('')

    return x_ticks_values


fig1, ax1 = plt.subplots(figsize=(21,12))
x_val_1 = plot_set_of_values('Próby wątrobowe', ['ALT', 'AST', 'Bilirubina całkowita'], [], ax1)
fig1.subplots_adjust(top=0.965,
                    bottom=0.08,
                    left=0.075,
                    right=0.895,
                    hspace=0.075,
                    wspace=0.185)

fig2, ax2 = plt.subplots(nrows=2, figsize=(21,12))
x_val_1 = plot_set_of_values('Cholesterol',
                             ['Cholesterol całkowity', 'Cholesterol HDL',
                              'Cholesterol nie-HDL', 'Cholesterol LDL', 'Trójglicerydy'], [],
                             ax2[0], False)
x_val_1 = plot_set_of_values('Ekhe stosunki :)', ['Całkowity do HDL', 'LDL do HDL', 'Trójglicerydy do HDL'],
                             [r'$\frac{Cholesterol całkowity}{Cholesterol HDL}$',
                              r'$\frac{Cholesterol LDL}{Cholesterol HDL}$',
                              r'$\frac{Trójglicerydy}{Cholesterol HDL}$'], ax2[1])
fig2.subplots_adjust(top=0.965,
                    bottom=0.08,
                    left=0.075,
                    right=0.895,
                    hspace=0.075,
                    wspace=0.185)


fig3, ax3 = plt.subplots(figsize=(21,12))
x_val_1 = plot_set_of_values('Cukry', ['Glukoza'], [], ax3)

fig3.subplots_adjust(top=0.965,
                    bottom=0.08,
                    left=0.075,
                    right=0.895,
                    hspace=0.075,
                    wspace=0.185)

plt.show()

