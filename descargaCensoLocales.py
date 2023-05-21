#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import requests
import pandas as pd 
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')


# # Censo de locales de Madrid
# 
# Utilizamos el csv obtenido del Ayuntamiento de Madrid con el censo de los locales de Madrid. Comenzamos leyendo el fichero para su tratamiento.

# In[2]:


df_total = pd.read_csv('datos/raw/datosCensoLocalesActividades.csv',sep=';')


# Se realiza un tratamiento del ID del barrio puesto que el formato que utiliza es distinto al que se dispone en el resto de ficheros. Es por esto que es necesario hacer un pequeño procesamiento del campo "id_barrio_local".

# In[10]:


df_total['id_barrio_local'] = df_total['id_barrio_local'].str.replace('0','')
df_total


# Obtenemos el número de locales que hay por cada barrio. Para ello hacemos un group by por el id de cada barrio y el año y sumamos el numero de locales. Posteroprmente renombramos las columnas obtenidas y almacenamos el resultado en un csv.

# In[11]:


df_total = df_total[['id_distrito_local','desc_distrito_local', 'id_barrio_local','cod_barrio_local', 'desc_barrio_local','ANIO', 'id_epigrafe', 'desc_epigrafe', 'desc_division' ]]
df_final = df_total.groupby(['id_barrio_local','ANIO'], as_index=False)['desc_barrio_local'].count()
df_final.rename(columns={"desc_barrio_local":"numero_locales", "id_barrio_local":"CodigoBarrio", "ANIO":"Anio"}, inplace=True)
df_final.to_csv('datos/raw/datosCensoLocalesActividadesNumero.csv',sep=';',index=False) 
df_final


# Realizamos un proceso similar al anterior para obtener el número de alojamientos de cada barrio. En este caso se debe filtrar del total del censo, aquellos locales cuyo epigrafe se encuentra entre “HOSTELERIA”, “SERVICIOS DE ALOJAMIENTO” Y “VIVIENDAS TURÍSTICAS”, que son los que se registran como alojamientos turisticos.

# In[12]:


#Buscamos los epigrafes  “HOSTELERIA”, “SERVICIOS DE ALOJAMIENTO” Y “VIVIENDAS TURÍSTICAS” que se corresponden con alojamientos turisticos

epigrafes = ["HOSTELERIA", "SERVICIOS DE ALOJAMIENTO","VIVIENDAS TURÍSTICAS"]
df_alojamientos = df_total[df_total['desc_division'].isin(epigrafes)]
df_alojamientos = df_alojamientos.groupby(['id_barrio_local','ANIO'], as_index=False)['desc_epigrafe'].count()
df_alojamientos.rename(columns={"desc_epigrafe":"numero_alojamientos", "id_barrio_local":"CodigoBarrio","ANIO":"Anio"}, inplace=True)
df_alojamientos.to_csv('datos/raw/datosCensoLocalesAlojamientoNumero.csv',sep=';',index=False) 

df_alojamientos

