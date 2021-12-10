import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import matplotlib.patches as mpatches
import os
import xml.etree.ElementTree as et

sns.set(style='darkgrid')


dir_name = r'data_real'
fileList = os.listdir(dir_name)

df = None
df_alab = pd.DataFrame(columns=['Parametr', 'Materiał', 'Data', 'Wynik', 'Jednostka', 'Min', 'Max'])

for fileItem in fileList:
    fileName = os.path.join('.\\'+dir_name, fileItem)

    if fileItem.split('.')[-1].lower() == 'csv':
        if df is None:
            df = pd.read_csv(fileName, sep=';', decimal=",")
        else:
            df1 = pd.read_csv(fileName, sep=';', decimal=",")
            df = pd.concat([df, df1], axis=0)

    elif fileItem.split('.')[-1].lower() == 'xml':
        file_root = et.parse(fileName).getroot()
        for node in file_root:  # look for all Formularze
            for formularz in node:  # look for all Formularz
                badanie_data = formularz.find('Zlecenie').find('Data').text
                sekcje = formularz.find('Sekcje')
                for sekcja_node in sekcje:
                    probki_node = sekcja_node.find('Próbki')
                    for probka in probki_node:
                        badanie_material = probka.find('Materiał').find('Symbol').text
                        for wykonianie in probka.find('Wykonania'):
                            for wyniki in wykonianie.find('Wyniki'):
                                badanie_nazwa = wyniki.find('Parametr').find('Nazwa').text
                                badanie_symbol = wyniki.find('Parametr').find('Symbol').text
                                if wyniki.find('Parametr').find('Jednostka') is not None:
                                    badanie_jednostka = wyniki.find('Parametr').find('Jednostka').text
                                else:
                                    badanie_jednostka = "--"
                                if wyniki.find('WynikLiczbowy') is not None:
                                    badanie_wynik = wyniki.find('WynikLiczbowy').text
                                    badanie_wynik_txt = ""
                                else:
                                    badanie_wynik = np.nan
                                    if wyniki.find('WynikTekstowy') is not None:
                                        badanie_wynik_txt = wyniki.find('WynikTekstowy').text
                                    else:
                                        badanie_wynik_txt = ""
                                if wyniki.find('Norma') is not None:
                                    if wyniki.find('Norma').find('ZakresOd') is not None:
                                        badanie_zakres_od = wyniki.find('Norma').find('ZakresOd').text
                                    else:
                                        badanie_zakres_od = 0
                                    if wyniki.find('Norma').find('ZakresDo') is not None:
                                        badanie_zakres_do = wyniki.find('Norma').find('ZakresDo').text
                                    else:
                                        badanie_zakres_do = np.inf
                                else:
                                    badanie_zakres_od = 0
                                    badanie_zakres_do = np.inf

                                temp_df = pd.DataFrame([[badanie_nazwa, badanie_material, badanie_data,
                                                        badanie_wynik, badanie_jednostka,
                                                         badanie_zakres_od, badanie_zakres_do]],
                                                       columns=['Parametr', 'Materiał', 'Data',
                                                                'Wynik', 'Jednostka',
                                                                'Min', 'Max'])

                                df_alab = df_alab.append(temp_df, ignore_index=True)

#print(df_alab.loc[:, 'Materiał'].unique())

df_alab.loc[df_alab.loc[:,'Materiał'] != 'MOCZ','Materiał'] = 'KREW'
df_alab.loc[df_alab.loc[:,'Materiał'] == 'MOCZ','Materiał'] = 'MOCZ'

df_alab.loc[:, 'Lab'] = 'ALAB'
df.loc[:, 'Lab'] = 'Diag'

#pd.set_option('display.expand_frame_repr', False)
#print(df_alab)
#pd.set_option('display.expand_frame_repr', True)

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

df.loc[:, 'Materiał'] = ''
df.loc[df.loc[:, 'Badanie'] != 'Badanie ogólne moczu', 'Materiał'] = 'KREW'
df.loc[df.loc[:, 'Badanie'] == 'Badanie ogólne moczu', 'Materiał'] = 'MOCZ'

df = df.drop(labels=['Badanie', 'Kod zlecenia', 'Zakres referencyjny', 'Opis'], axis=1)

print(df.columns)

df.loc[:, 'Data'] = pd.to_datetime(df.loc[:, 'Data'], format='%d-%m-%Y %H:%M:%S')
df_alab.loc[:, 'Data'] = pd.to_datetime(df_alab.loc[:, 'Data'], format='%Y-%m-%d')

df.loc[df['Min'] == '', 'Min'] = "0,0"
df.loc[df['Max'] == '', 'Max'] = str(np.inf)

df.loc[:, 'Wynik'] = df.loc[:, 'Wynik'].str.replace(",", ".").astype(float)
df.loc[:, 'Min'] = df.loc[:, 'Min'].str.replace(",", ".").astype(float)
df.loc[:, 'Max'] = df.loc[:, 'Max'].str.replace(",", ".").astype(float)

#match names
if (df_alab is not None):
    df_alab.loc[:, 'Wynik'] = df_alab.loc[:, 'Wynik'].astype(float)
    df_alab.loc[:, 'Min'] = df_alab.loc[:, 'Min'].astype(float)
    df_alab.loc[:, 'Max'] = df_alab.loc[:, 'Max'].astype(float)

    # match names

    df_alab.loc[df_alab.loc[:,'Parametr'] == 'Odczyn Biernackiego (C59)', 'Parametr'] = 'OB'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Triglicerydy (O49)', 'Parametr'] = 'Trójglicerydy'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Kwas moczowy  w surowicy (M45)', 'Parametr'] = 'Kwas moczowy'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Glukoza (L43)', 'Parametr'] = 'Glukoza'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'HDL-cholesterol (K01)', 'Parametr'] = 'Cholesterol HDL'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Cholesterol całkowity (I99)', 'Parametr'] = 'Cholesterol całkowity'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Nie-HDL', 'Parametr'] = 'Cholesterol nie-HDL'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Cholesterol LDL - wyliczany (K03)', 'Parametr'] = 'Cholesterol LDL'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Bilirubina całkowita (I89)', 'Parametr'] = 'Bilirubina całkowita'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Aminotransferaza alaninowa (ALT) (I17)', 'Parametr'] = 'ALT'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Kreatynina (M37)', 'Parametr'] = 'Kreatynina'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Mocznik (N13)', 'Parametr'] = 'Mocznik'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Białko C-reaktywne (CRP) - ilościowe (I81)', 'Parametr'] = 'CRP'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'HBs - antygen HBs (WZW typu B) (V39)', 'Parametr'] = 'Przeciwciała anty HBs'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Potas w surowicy (N45)', 'Parametr'] = 'Potas'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Tyreotropina (TSH)  trzeciej generacji (L69)', 'Parametr'] = 'TSH'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Gamma-glutamylotranspeptydaza (GGTP) (L31)', 'Parametr'] = 'GGTP'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Aminotransferaza asparaginianowa (AST) (I19)', 'Parametr'] = 'AST'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Sód w surowicy (O35)', 'Parametr'] = 'Sód'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'PH', 'Parametr'] = 'pH'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Eozynofile %', 'Parametr'] = 'Eozynofile'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Bazofile %', 'Parametr'] = 'Bazofile'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'Monocyty %', 'Parametr'] = 'Monocyty'
    #df_alab.loc[df_alab.loc[:, 'Parametr'] == 'LUC', 'Parametr'] = 'Leukocyty'
    #df_alab.loc[df_alab.loc[:, 'Parametr'] == 'LUC %', 'Parametr'] = 'Leukocyty'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'EGFR wyliczane z MDRD (M37)', 'Parametr'] = 'eGFR'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'RDW', 'Parametr'] = 'RDW-CV'
    df_alab.loc[df_alab.loc[:, 'Parametr'] == 'PŁYTKI', 'Parametr'] = 'Płytki krwi'

pd.set_option('display.expand_frame_repr', False)
print(df)
print(df_alab)

df = df.append(df_alab)

print(df)

pd.set_option('display.expand_frame_repr', True)

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
    #plt.fill_between(x='Data', y1='Min', y2='Max', data=curr_data_local)

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


fig1, ax1 = plt.subplots(figsize=(21, 12))
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

