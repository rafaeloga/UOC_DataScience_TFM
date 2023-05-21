#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import requests
import pandas as pd 
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')


# ## Merge de datasets de población
# 
# Cargamos los datos desde disco para evitar decargar todos los ficheros:

# In[3]:


from functools import reduce

df_poblacion = pd.read_csv('datos/raw/poblacionPorBarrio.csv', decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoDistrito','NombreDistrito','CodigoBarrio','NombreBarrio','DatoPoblacion']]  
df_poblacionMenor16 = pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeMenor16.csv', decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionMenor16']]
df_poblacionEntre16y64 = pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeEntre1664.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionEntre16y64']]
df_poblacionMas65 =  pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeMas65.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionMayor65']]
df_sobreenv = pd.read_csv('datos/raw/poblacionIndiceSobreenvejecimiento.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoSobreEnvejecimiento']]
df_sobreenv_sexo = pd.read_csv('datos/raw/poblacionIndiceSobreenvejecimientoSexo.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoSobreenvejecimientoHombres','DatoSobreenvejecimientoMujeres']]
df_feminidad = pd.read_csv('datos/raw/poblacionIndiceFeminidad.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoFeminidad']]
df_extranjeros = pd.read_csv('datos/raw/poblacionNacionalidadExtranjera.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoNacionalidadExtranjera']]

dfs = [df_poblacion,df_poblacionMenor16,df_poblacionEntre16y64,df_poblacionMas65,df_sobreenv,df_sobreenv_sexo, df_feminidad,df_extranjeros]

df_result_poblacion = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='inner', on=['Anio','Periodo','CodigoBarrio']), 
                                              dfs)

df_result_poblacion.to_csv("datos/silver/result_datos_población.csv",sep=';',index=False) 
df_result_poblacion


# ## Merge datos economicos
# 
# 

# In[4]:



df_RentaMedia = pd.read_csv('datos/raw/economicosRentaMediaPorHogar.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoRentaMedia']]
df_ParadosRgistrados = pd.read_csv('datos/raw/economicosParadosRegistrados.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoParadosRegistrados']]
df_ParadosRgistrados_sexo = pd.read_csv('datos/raw/economicosParadosRegistradosSexo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio','DatoParadosRegistradosHombres','DatoParadosRegistradosMujer']]
df_AfiliacionesTrabajo = pd.read_csv('datos/raw/economicosAfiliacionesTrabajo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesTrabajo']]  
df_AfiliacionesTrabajo_sexo = pd.read_csv('datos/raw/economicosAfiliacionesTrabajoSexo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo', 'CodigoBarrio','DatoAfiliacionesTrabajoHombres','DatoAfiliacionesTrabajoMujer']]
df_AfiliacionesResidencia = pd.read_csv('datos/raw/economicosAfiliacionesReidencia.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesResidencia']]
df_AfiliacionesResidencia_sexo =  pd.read_csv('datos/raw/economicosAfiliacionesResidenciaSexo.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesResidenciaHombres','DatoAfiliacionesResidenciaMujer']]
df_Autonomos = pd.read_csv('datos/raw/economicosAutonomos.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoAutonomo']]
df_Autonomos_sexo = pd.read_csv('datos/raw/economicosAutonomosSexo.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo', 'CodigoBarrio','DatoAutonomosHombres','DatoAutonomosMujer']]



#Modificamos los dataset de parados para coger solamente los datos de Diciembre de cada año y cambiamos el periodo a anual:
df_ParadosRgistrados_anual = df_ParadosRgistrados[df_ParadosRgistrados['Periodo'] =='Diciembre']
df_ParadosRgistrados_anual['Periodo'] = 'Anual'

df_ParadosRgistrados_sexo_anual = df_ParadosRgistrados_sexo[df_ParadosRgistrados_sexo['Periodo'] =='Diciembre']
df_ParadosRgistrados_sexo_anual['Periodo'] = 'Anual'





dfs = [df_result_poblacion,
       df_ParadosRgistrados_anual,
       df_ParadosRgistrados_sexo_anual,
       df_AfiliacionesTrabajo,
       df_AfiliacionesTrabajo_sexo,
       df_AfiliacionesResidencia, 
       df_AfiliacionesResidencia_sexo,
       df_Autonomos,
       df_Autonomos_sexo]

df_result_poblacion_economico = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='outer', on=['Anio','Periodo','CodigoBarrio']), 
                                              dfs)


df_result_poblacion_economico.to_csv("datos/silver/result_datos_poblacion_economicos.csv",sep=';',index=False) 


df_result_poblacion_economico


# ## Merge datos de Idealista

# In[5]:


df_datos_poblacion_economicos = pd.read_csv('datos/silver/result_datos_poblacion_economicos.csv', decimal=',', delimiter=';',header=0)

df_precio_alquiler = pd.read_csv('datos/raw/precioalquilerhistorico.csv', decimal=',', delimiter=';',header=0)[['ANIO','CODDIS', 'PrecioAlquilerEurosM2']]
df_precio_alquiler.columns = ['Anio', 'CodigoDistrito', 'PrecioAlquilerEurosM2']

df_precio_venta = pd.read_csv('datos/raw/precioventahistorico.csv', decimal=',', delimiter=';',header=0)[['ANIO','CODDIS', 'PrecioVentaEurosM2']]
df_precio_venta.columns = ['Anio', 'CodigoDistrito', 'PrecioVentaEurosM2']

dfs = [df_datos_poblacion_economicos,
       df_precio_venta,
       df_precio_alquiler]

df_result_poblacion_economico_idealista = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='outer', on=['Anio','CodigoDistrito']), 
                                              dfs)


df_result_poblacion_economico_idealista.to_csv("datos/silver/result_datos_poblacion_economicos_idealista.csv",sep=';',index=False) 

df_result_poblacion_economico_idealista


# ## Merge dataset de Inversiones
# 

# In[6]:


df_datos_poblacion_economicos_idealista =  pd.read_csv('datos/silver/result_datos_poblacion_economicos_idealista.csv', decimal=',', delimiter=';',header=0)

df_datos_inversiones = pd.read_csv("datos/raw/inversiones-madrid.csv",decimal='.', delimiter=',',header=0)
df_datos_inversiones.columns = ['Anio','CodigoDistrito','NombreDistrito','PresupuestoGasto','Gasto_Real']

df_datos_inversiones = df_datos_inversiones[df_datos_inversiones['CodigoDistrito'] != 'NN'].dropna()

df_datos_inversiones['CodigoDistrito'] = (df_datos_inversiones['CodigoDistrito'].astype('int') - 200)

df_datos_poblacion_economicos_idealista['CodigoBarrio'] = (df_datos_poblacion_economicos_idealista['CodigoBarrio'].astype('str'))

dfs = [df_datos_poblacion_economicos_idealista,
       df_datos_inversiones]

df_result_poblacion_economico_idealista_inversiones = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='outer', on=['Anio','CodigoDistrito']), 
                                              dfs)

df_result_poblacion_economico_idealista_inversiones = df_result_poblacion_economico_idealista_inversiones[df_result_poblacion_economico_idealista_inversiones['Anio'] != 2023]

df_result_poblacion_economico_idealista_inversiones = df_result_poblacion_economico_idealista_inversiones.drop('NombreDistrito_y', axis=1)
df_result_poblacion_economico_idealista_inversiones.rename(columns={'NombreDistrito_x':'NombreDistrito'}, inplace=True)


df_result_poblacion_economico_idealista_inversiones.to_csv("datos/silver/result_datos_poblacion_economicos_idealista_inversiones.csv",sep=';',index=False) 

df_result_poblacion_economico_idealista_inversiones


# ## Merge del Censo de locales

# In[7]:


df_result_poblacion_economico_idealista_inversiones =  pd.read_csv('datos/silver/result_datos_poblacion_economicos_idealista_inversiones.csv', decimal=',', delimiter=';',header=0)

df_result_poblacion_economico_idealista_inversiones['CodigoDistrito'] = (df_result_poblacion_economico_idealista_inversiones['CodigoDistrito'].astype('str'))

df_datos_censo_locales = pd.read_csv('datos/raw/datosCensoLocalesActividadesNumero.csv', decimal=',', delimiter=';',header=0)
df_datos_censo_locales.columns=['CodigoBarrio','Anio','NumeroLocales']



df_datos_censo_alojamientos = pd.read_csv('datos/raw/datosCensoLocalesAlojamientoNumero.csv', decimal=',', delimiter=';',header=0)
df_datos_censo_alojamientos.columns=['CodigoBarrio','Anio','NumeroAlojamientos']

dfs = [df_result_poblacion_economico_idealista_inversiones,
       df_datos_censo_locales, df_datos_censo_alojamientos]

df_result_poblacion_economico_idealista_inversiones_locales = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='outer', on=['Anio','CodigoBarrio']), 
                                              dfs)

df_result_poblacion_economico_idealista_inversiones_locales.dropna(subset=['CodigoDistrito'], inplace=True)
df_result_poblacion_economico_idealista_inversiones_locales['CodigoDistrito'] = (df_result_poblacion_economico_idealista_inversiones_locales['CodigoDistrito'].astype('str'))

df_result_poblacion_economico_idealista_inversiones_locales.to_csv("datos/silver/result_datos_poblacion_economicos_idealista_inversiones_locales.csv",sep=';',index=False) 

df_result_poblacion_economico_idealista_inversiones_locales


# In[8]:


df_result = pd.merge(df_result_poblacion_economico_idealista_inversiones_locales, df_RentaMedia, how='inner', on=['Anio','CodigoBarrio'])
df_result.to_csv("datos/silver/result_datos_poblacion_economicos_idealista_inversiones_locales_RENTA.csv",sep=';',index=False)

