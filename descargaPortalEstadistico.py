#!/usr/bin/env python
# coding: utf-8

# # Imporación de librerias.
# 
# Se realiza la importación de las librerias necesarias para la ejecución de todos el código.

# In[1]:


import csv
import requests
import pandas as pd 
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')


# Se crea la función downloadInfo mediante la cual se le indica la información que se quiere descargar, la url donde se quiere almacenar y los datos que se quieren guardar.

# In[2]:



def donwloadInfo(baseURL, completeURL, columnas=['Anio', 'Periodo', 'Dato']):
    df_complete = pd.DataFrame()

    for index, barrio in barrios_df.iterrows():
           
            cod_barrio = barrio['COD_BAR']
            cod_distrito = barrio['CODDIS']
            nombre_barrio = barrio['NOMBRE']
            nombre_distrito = barrio['NOMDIS']

            print("Descargando BAR"  + str(cod_barrio))
            if cod_barrio > 100:
                url = baseURL.replace('BAR0X', 'BAR' + str(cod_barrio))
            else:
                url = baseURL.replace('BAR0X', 'BAR0' + str(cod_barrio))
            print("Url: " + url)
            df = pd.read_csv(url, delimiter=';', decimal=',', encoding='utf-8', on_bad_lines='skip',index_col=False, header=2, encoding_errors='ignore', names=columnas)
            df['CodigoBarrio'] = cod_barrio
            df['CodigoDistrito'] = cod_distrito
            df['NombreBarrio'] = nombre_barrio
            df['NombreDistrito'] = nombre_distrito

            df_complete = pd.concat([df,df_complete], axis=0, ignore_index=False)
            print()
            clear_output(wait=True)

    df_complete.to_csv(completeURL,sep=';',index=False) 
    return(df_complete) 


# ## Carga de datos de Barrios de Madrid
# 
# Se lee el csv con los datos de los Barrios de Madrid para obtener un catalogo inicial de los barrios de Madrid. Se obtiene tanto el codigo de barrio, el nombre de barrio asi como el codigo de distrito y nombre del distrito al que pertenece cada barrio.

# In[3]:




barrios_df = pd.read_csv('datos/raw/Barrios.csv', decimal=',', delimiter=';',header=0)[['COD_BAR', 'NOMBRE', 'CODDIS','NOMDIS']]                                                                                        
barrios_df.sort_values('COD_BAR')


# ## Datos de población por barrio
# 
# Personas empadronadas a 1 de enero de cada año, salvo 1991 que es a 1 de marzo y 1996 que es a 1 de mayo. Los datos comparativos van referidos al último año para el que existe información para todos los territorios.

# In[4]:


URL_POBLACION = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ZTV21&idp=35&Id_Celda_Fila_Plantilla=8373&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

df_poblacion= donwloadInfo(URL_POBLACION, 'datos/raw/poblacionPorBarrio.csv',['Anio','Periodo', 'DatoPoblacion'])
df_poblacion


# ## Porcentaje de población de menos de 16 años
# 
# Porcentaje de población menor de 16 años sobre la población total. Los datos comparativos van referidos al último año para el que existe información para todos los territorios.

# In[5]:


URL_POBLACION_MENOR_16 = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8956&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8956&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv


df_poblacionMenor16= donwloadInfo(URL_POBLACION_MENOR_16, 'datos/raw/poblacionPorBarrioPOrcetajeMenor16.csv', ['Anio','Periodo', 'DatoPoblacionMenor16'])
df_poblacionMenor16


# ## Porcentaje de población de 16 a 64 años
# 
# Porcentaje de población de 16 a 64 años sobre la población total. Los datos comparativos van referidos al último año para el que existe información para todos los territorios.

# In[6]:


URL_POBLACION_ENTRE_16_64 = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8958&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv1&idp=34&Id_Celda_Fila_Plantilla=8958&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv


df_poblacionEntre16y64= donwloadInfo(URL_POBLACION_ENTRE_16_64, 'datos/raw/poblacionPorBarrioPOrcetajeEntre1664.csv', ['Anio','Periodo', 'DatoPoblacionEntre16y64'])
df_poblacionEntre16y64


# ## Porcentaje de población de 65 y más años
# 
# 
# Porcentaje de población de 65 y más años sobre la población total. Los datos comparativos van referidos al último año para el que existe información para todos los territorios.

# In[7]:


URL_POBLACION_MAS_65 = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8960&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8960&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv


df_poblacionMas65= donwloadInfo(URL_POBLACION_MAS_65, 'datos/raw/poblacionPorBarrioPOrcetajeMas65.csv', ['Anio','Periodo', 'DatoPoblacionMayor65'])
df_poblacionMas65


# ## Índice de sobreenvejecimiento
# 
# Porcentaje de población mayor de 79 años sobre la población mayor de 64 años.
# 
# Periodo:	Anual 2022
# 
# Fuente:	Ciudad de Madrid: Ayuntamiento de Madrid, Explotación del Padrón
# 
# Cálculo:	(población de 80 años y más / población de 65 años y más) x 100
# 
# Porcentaje de población mayor de 79 años sobre la población mayor de 64 años.

# In[8]:


URL_SOBREENVEJICIMIENTO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8962&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

df_sobreenv= donwloadInfo(URL_SOBREENVEJICIMIENTO, 'datos/raw/poblacionIndiceSobreenvejecimiento.csv', ['Anio','Periodo', 'DatoSobreEnvejecimiento'])
df_sobreenv


# In[9]:


URL_SOBREENVEJICIMIENTO_SEXO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8963&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-39&idp=34&Id_Celda_Fila_Plantilla=8963&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_sobreenv_sexo= donwloadInfo(URL_SOBREENVEJICIMIENTO_SEXO, 'datos/raw/poblacionIndiceSobreenvejecimientoSexo.csv', ['Anio', 'Periodo', 'DatoSobreenvejecimientoHombres', 'DatoSobreenvejecimientoMujeres'])
df_sobreenv_sexo


# ## Índice de feminidad
# 
# Número de mujeres por cada 100 hombres
# 
# Periodo:	Anual 2022
# 
# Fuente:	Ciudad de Madrid: Ayuntamiento de Madrid, Explotación del Padrón
# 
# Cálculo:	(población de mujeres de 65 y más años / población de hombres de 65 y más años) x 100
# 
# Número de mujeres por cada 100 hombres

# In[10]:


URL_FEMINIDAD = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8964&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-39&idp=34&Id_Celda_Fila_Plantilla=8964&Id_Territorio=28079BAR172&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_feminidad= donwloadInfo(URL_FEMINIDAD, 'datos/raw/poblacionIndiceFeminidad.csv',['Anio','Periodo', 'DatoFeminidad'])
df_feminidad


# ## Porcentaje de población de nacionalidad extranjera
# 
# Periodo:	Anual 2022
# 
# Fuente:	Ciudad de Madrid: Ayuntamiento de Madrid, Explotación del Padrón; Resto de municipios: INE, Revisión del Padrón.
# 
# Cálculo:	(población extranjera / población total)*100
# 
# Porcentaje de población de nacionalidad extranjera sobre el total de la población. Los datos comparativos van referidos al último año para el que existe información para todos los territorios.

# In[11]:


URL_NACIONALIDAD_EXTRANJERA = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8966&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8966&Id_Territorio=28079BAR172&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_NacionalidadEstranjera= donwloadInfo(URL_NACIONALIDAD_EXTRANJERA, 'datos/raw/poblacionNacionalidadExtranjera.csv',['Anio','Periodo', 'DatoNacionalidadExtranjera'])
df_NacionalidadEstranjera


# ## Merge de datasets de población
# 
# Cargamos los datos desde disco para evitar decargar todos los ficheros:

# In[12]:


from functools import reduce

df_poblacion = pd.read_csv('datos/raw/poblacionPorBarrio.csv', decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoDistrito','CodigoBarrio','NombreBarrio','DatoPoblacion']]  
df_poblacionMenor16 = pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeMenor16.csv', decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionMenor16']]
df_poblacionEntre16y64 = pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeEntre1664.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionEntre16y64']]
df_poblacionMas65 =  pd.read_csv('datos/raw/poblacionPorBarrioPOrcetajeMas65.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoPoblacionMayor65']]
df_sobreenv = pd.read_csv('datos/raw/poblacionIndiceSobreenvejecimiento.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoSobreEnvejecimiento']]
df_sobreenv_sexo = pd.read_csv('datos/raw/poblacionIndiceSobreenvejecimientoSexo.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoSobreenvejecimientoHombres','DatoSobreenvejecimientoMujeres']]
df_feminidad = pd.read_csv('datos/raw/poblacionIndiceFeminidad.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoFeminidad']]

dfs = [df_poblacion,df_poblacionMenor16,df_poblacionEntre16y64,df_poblacionMas65,df_sobreenv,df_sobreenv_sexo, df_feminidad]

df_result_poblacion = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              how='inner', on=['Anio','Periodo','CodigoBarrio']), 
                                              dfs)
df_result_poblacion


# ## Parados Registrados
# Demandantes de empleo registrados en las oficinas del SEPE el último día de cada mes.

# In[13]:


URL_PARADOS_REGISTRADOS = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8968&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8968&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_ParadosRgistrados= donwloadInfo(URL_PARADOS_REGISTRADOS, 'datos/raw/economicosParadosRegistrados.csv',['Anio','Periodo', 'DatoParadosRegistrados'])
df_ParadosRgistrados


# In[14]:


df_ParadosRgistrados_anual = df_ParadosRgistrados[df_ParadosRgistrados['Periodo'] =='Diciembre']
df_ParadosRgistrados_anual['Periodo'] = 'Anual'

df_ParadosRgistrados_anual


# In[15]:


URL_PARADOS_REGISTRADOS_SEXO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_ParadosRgistrados_sexo= donwloadInfo(URL_PARADOS_REGISTRADOS_SEXO, 'datos/raw/economicosParadosRegistradosSexo.csv',['Anio','Periodo', 'DatoParadosRegistradosHombres','DatoParadosRegistradosMujer'])
df_ParadosRgistrados_sexo


# ## Afiliaciones a la Seguridad Social por lugar de trabajo
# Número de afiliaciones por lugar de de trabajo, con alta laboral en la Seguridad Social, el último día del mes (dato mensual) o el 1 de enero (dato anual)

# In[16]:


URL_AFILIACIONES_TRABAJO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8974&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8968&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_AfiliacionesTrabajo= donwloadInfo(URL_AFILIACIONES_TRABAJO, 'datos/raw/economicosAfiliacionesTrabajo.csv',['Anio','Periodo', 'DatoAfiliacionesTrabajo'])
df_AfiliacionesTrabajo


# In[17]:


URL_AFILIACIONES_TRABAJO_SEXO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8975&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_AfiliacionesTrabajo_sexo= donwloadInfo(URL_AFILIACIONES_TRABAJO_SEXO, 'datos/raw/economicosAfiliacionesTrabajoSexo.csv',['Anio','Periodo', 'DatoAfiliacionesTrabajoHombres','DatoAfiliacionesTrabajoMujer'])
df_AfiliacionesTrabajo_sexo


# ## Afiliaciones de residentes a la Seguridad Social

# In[18]:


URL_AFILIACIONES_RESIDENCIA = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8985&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8968&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_AfiliacionesResidencia= donwloadInfo(URL_AFILIACIONES_RESIDENCIA, 'datos/raw/economicosAfiliacionesReidencia.csv',['Anio','Periodo', 'DatoAfiliacionesResidencia'])
df_AfiliacionesResidencia


# In[19]:


URL_AFILIACIONES_RESIDENCIA_SEXO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8986&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_AfiliacionesResidencia_sexo= donwloadInfo(URL_AFILIACIONES_RESIDENCIA_SEXO, 'datos/raw/economicosAfiliacionesResidenciaSexo.csv',['Anio','Periodo', 'DatoAfiliacionesResidenciaHombres','DatoAfiliacionesResidenciaMujer'])
df_AfiliacionesResidencia_sexo


# ## Afiliaciones por lugar de trabajo al régimen especial de trabajadores autónomos

# In[20]:


URL_AUTONOMOS = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8979&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8968&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_Autonomos= donwloadInfo(URL_AUTONOMOS, 'datos/raw/economicosAutonomos.csv',['Anio','Periodo', 'DatoAutonomo'])
df_Autonomos


# In[21]:


URL_AUTONOMOS_SEXO = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8980&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_Autonomos_sexo= donwloadInfo(URL_AUTONOMOS_SEXO, 'datos/raw/economicosAutonomosSexo.csv',['Anio','Periodo', 'DatoAutonomosHombres','DatoAutonomosMujer'])
df_Autonomos_sexo


# ## Merge datos economicos
# 
# 

# In[22]:



df_ParadosRgistrados = pd.read_csv('datos/raw/economicosParadosRegistrados.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoParadosRegistrados']]
df_ParadosRgistrados_sexo = pd.read_csv('datos/raw/economicosParadosRegistradosSexo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio','DatoParadosRegistradosHombres','DatoParadosRegistradosMujer']]
df_AfiliacionesTrabajo = pd.read_csv('datos/raw/economicosAfiliacionesTrabajo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesTrabajo']]  
df_AfiliacionesTrabajo_sexo = pd.read_csv('datos/raw/economicosAfiliacionesTrabajoSexo.csv', decimal=',', delimiter=';',header=0)[['Anio','Periodo', 'CodigoBarrio','DatoAfiliacionesTrabajoHombres','DatoAfiliacionesTrabajoMujer']]
df_AfiliacionesResidencia = pd.read_csv('datos/raw/economicosAfiliacionesReidencia.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesResidencia']]
df_AfiliacionesResidencia_sexo =  pd.read_csv('datos/raw/economicosAfiliacionesResidenciaSexo.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo','CodigoBarrio', 'DatoAfiliacionesResidenciaHombres','DatoAfiliacionesResidenciaMujer']]
df_Autonomos = pd.read_csv('datos/raw/economicosAutonomos.csv',decimal=',', delimiter=';',header=0)[['Anio', 'Periodo','CodigoBarrio','DatoAutonomo']]
df_Autonomos_sexo = pd.read_csv('datos/raw/economicosAutonomosSexo.csv',decimal=',', delimiter=';',header=0)[['Anio','Periodo', 'CodigoBarrio','DatoAutonomosHombres','DatoAutonomosMujer']]



#Modificamos los dataset de a planifica
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
                                              how='inner', on=['Anio','Periodo','CodigoBarrio']), 
                                              dfs)
df_result_poblacion_economico


# ## Renta neta media por hogar

# In[23]:


URL_RENTA_MEDIA = "http://portalestadistico.com/municipioencifras/proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv21&idp=34&Id_Celda_Fila_Plantilla=8981&Id_Territorio=28079BAR0X&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv"

#proceso_descarga_excel_csv.aspx?pn=madrid&pc=ztv-49&idp=34&Id_Celda_Fila_Plantilla=8969&Id_Territorio=28079BAR016&Id_Territorio_Padre=28079&idioma=1&Tipo_Fichero_Generado=csv

df_renta_media= donwloadInfo(URL_RENTA_MEDIA, 'datos/raw/economicosRentaMediaPorHogar.csv',['Anio','Periodo','DatoRentaMedia'])
df_renta_media

