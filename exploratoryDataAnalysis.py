#!/usr/bin/env python
# coding: utf-8

# ## Limpieza de datos del dataset
# 
# Una vez se tiene el dataset que sera objeto de estudio, y antes de realizar ningún analisis de los datos, es necesario realizar la limpieza de los datos. 
# 
# Importamos las librerias que serán secesarias tanto para la limpieza como para el Análisis exploratorio.
# 

# In[2]:


# data manipulation
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# aplicamos algunas reglas de estilo para la visualización
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12, 6)


# Cargamos nuestro dataset y analizamos ciertos datos de interes que nos van a permitir conocer mejor los datos de los que disponemos.

# In[3]:


df_result_poblacion_economico_idealista_inversiones_locales =  pd.read_csv('datos/silver/result_datos_poblacion_economicos_idealista_inversiones_locales.csv', decimal=',', delimiter=';',header=0)


# Mostramos el tamaño del dataset con la función shape. 

# In[4]:


df_result_poblacion_economico_idealista_inversiones_locales.shape


# Esta función nos indica que disponemos de 2310 filas y 32 columnas. Vamos a obtener información con la funcion info:

# In[5]:


df_result_poblacion_economico_idealista_inversiones_locales.info()


# Es importante, en este punto, destacar que no existe ninguna columna repretida o que semanticamente tenga pueda tener el mismo significado. Además, debido a que el dataset se ha preparado con datos disponibles, podemos garantizar que todas las variables son utiles y tiene un sentido que se utilicen para posteriores estudios. Por eso se va a mantener el conjunto de columnas completo.
# 
# Los nombres de las variables son descriptivos por lo que tambien se mantienen.
# 
# A continuación, una vez comprobadas las columnas, pasamos a comprobar que no exista ninguna fila duplicada.

# In[6]:


df_result_poblacion_economico_idealista_inversiones_locales.duplicated().sum()


# Se puede ver que no exste ninguna fila duplicada por lo que no es necesario borrar ninguna fila.
# 
# Para comprobar que campos contienen valores nulos utilizamos la función describe.

# In[7]:


df_result_poblacion_economico_idealista_inversiones_locales.isna().sum()


# En este caso se tiene bastantes filas que contienen valores nulos en alguna de sus columnas. Como se explica anteriormente todos estos valores nulos provienen de los merge realizados con el metodo "outer" debido a que se cruzan periodos que no existen en todos los dataset o al hacer el cruce por distrito en lugar de barrio. Por esto tomamos la decisión de, de inicio, mantener todos los valores nulos. Posteriormente, segun sea necesario, se haran los filtrados necesarios para los analisis que lo necesiten.

# Se ha detectado que en las columnas DatoParadosRegistradosHombres y DatoParadosRegistradosMujer existen algunos registros que contienen el caracter '-'. Esto supone un problema debido a que cuando sea necesario tratar las variables como numericas python no será capaz de transformar esos regisros a numericos. Se decide, por tanto, sustituir dichos valores por un valor nulo o NaN

# In[8]:


df_result_poblacion_economico_idealista_inversiones_locales.loc[df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosHombres==' - ','DatoParadosRegistradosHombres']=np.NaN
df_result_poblacion_economico_idealista_inversiones_locales.loc[df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosMujer==' - ','DatoParadosRegistradosMujer']=np.NaN


# ## Análisis Exploratorio de los Datos
# 
# Una vez se ha realizado una limpieza de los datos, es el momento de realizar un analsis exploratorio completo de nuestras variables. De esta manera es posible tener un conocimiento del dataset. Para ello se comienza diferenciando las variables categoricas de las variables numericas o cuantitativas.
# 
# Entre las variables categoricas se pueden encontrar las siguiente:
# 
#     - Anio
#     - Periodo
#     - CodigoDistrito
#     - NombreDistrito
#     - CodigoBarrio
#     - NombreBarrio
# 
# Mientras el resto de variables son variables cuantitativas. Son las siguientes:
# 
#     - DatoPoblacion
#     - DatoPoblacionMenor16
#     - DatoPoblacionEntre16y64
#     - DatoPoblacionMayor65
#     - DatoSobreEnvejecimiento
#     - DatoSobreenvejecimientoHombres
#     - DatoSobreenvejecimientoMujeres
#     - DatoFeminidad
#     - DatoParadosRegistrados
#     - DatoParadosRegistradosHombres
#     - DatoParadosRegistradosMujer
#     - DatoAfiliacionesTrabajo
#     - DatoAfiliacionesTrabajoHombres
#     - DatoAfiliacionesTrabajoMujer
#     - DatoAfiliacionesResidencia
#     - DatoAfiliacionesResidenciaHombres
#     - DatoAfiliacionesResidenciaMujer
#     - DatoAutonomo
#     - DatoAutonomosHombres
#     - DatoAutonomosMujer
#     - PrecioVentaEurosM2
#     - PrecioAlquilerEurosM2
#     - PresupuestoGasto
#     - Gasto_Real
#     - NumeroLocales
#     - NumeroAlojamientos
# 
# A continuacion se realiza el estudio de las variables categoricas y se continua con las cuantitativas:
# 
# ### Variables categoricas:
# 
# #### Anio:
# Comprobamos el número de valores diferentes que tenemos en la variable con la función .value_counts().
# 

# In[9]:


df_result_poblacion_economico_idealista_inversiones_locales["Anio"].value_counts()


# Ademas, se puede expresar el resultado en porcentaje utilizando el parametro "normalize=True".

# In[10]:


df_result_poblacion_economico_idealista_inversiones_locales["Anio"].value_counts(normalize=True)


# Lo primero que se observa es que no todos los años tienen datos, lo cual se debe tener en cuenta en posteriores estudios, y, ademas, existe una entrada para anio = '0' que no debe existir. Se procede a borrarla.

# In[11]:


df_result_poblacion_economico_idealista_inversiones_locales = df_result_poblacion_economico_idealista_inversiones_locales.drop(
                            df_result_poblacion_economico_idealista_inversiones_locales[(df_result_poblacion_economico_idealista_inversiones_locales['Anio'] == 0)].index
                        )
df_result_poblacion_economico_idealista_inversiones_locales["Anio"].value_counts(normalize=True)


# Se obtiene una representación gráfica de los datos.

# In[12]:


plt.figure(figsize=(18,10))
fig = sns.countplot(x = 'Anio', data = df_result_poblacion_economico_idealista_inversiones_locales, palette=["#336699","#337799"])

plt.xlabel("Año")
plt.ylabel("Número de registros")
plt.title("Número de registros por año")

plt.show(fig)


# #### Periodo
# Se realiza el mismo analisis que para la variable anterior

# In[13]:


df_result_poblacion_economico_idealista_inversiones_locales["Periodo"].value_counts()


# Como se puede observar todos los registros contienen el valor "Anual". En este caso esta variable no aporta información puesto que todos los regsitros contienen el mismo valor. Esta variable, por tanto, no será utilizada en los futuros estudios.

# #### CodigoDistrito
# Se realiza el mismo analisis que anteriormente.

# In[14]:


df_result_poblacion_economico_idealista_inversiones_locales["CodigoDistrito"].value_counts()


# Se obtienen, de igual manera, los valores porcentuales. 

# In[15]:


df_result_poblacion_economico_idealista_inversiones_locales["CodigoDistrito"].value_counts(normalize=True)


# In[16]:


plt.figure(figsize=(18,10))
fig = sns.countplot(x = 'CodigoDistrito', data = df_result_poblacion_economico_idealista_inversiones_locales, palette=["#336699","#337799"])

plt.xlabel("Codigo Distrito")
plt.ylabel("Número de registros")
plt.title("Número de registros por Codigo de Distrito")

plt.show(fig)


# #### NombreDistrito

# Se realiza el mismo analisis para NombreDistrito

# In[17]:


df_result_poblacion_economico_idealista_inversiones_locales["NombreDistrito"].value_counts()


# In[18]:


df_result_poblacion_economico_idealista_inversiones_locales["NombreDistrito"].value_counts(normalize=True)


# In[19]:


plt.figure(figsize=(18,10))
fig = sns.countplot(x = 'NombreDistrito', data = df_result_poblacion_economico_idealista_inversiones_locales, 
                    palette=["#336699","#337799"])

plt.xlabel("Nombre Distrito")
plt.ylabel("Número de registros")
plt.title("Número de registros por Nombre de Distrito")
plt.xticks(rotation=90)
plt.show(fig)


# #### CodigoBarrio

# In[20]:


df_result_poblacion_economico_idealista_inversiones_locales["CodigoBarrio"].value_counts()


# In[21]:


df_result_poblacion_economico_idealista_inversiones_locales["CodigoBarrio"].value_counts(normalize=True)


# In[22]:


plt.figure(figsize=(35,25))
fig = sns.countplot(x = 'CodigoBarrio', data = df_result_poblacion_economico_idealista_inversiones_locales, 
                    palette=["#336699","#337799"])

plt.xlabel("Codigo Barrio")
plt.ylabel("Número de registros")
plt.title("Número de registros por Codigo de Barrio")
plt.xticks(rotation=90)
plt.show(fig)


# #### NombreBarrio

# In[23]:


df_result_poblacion_economico_idealista_inversiones_locales["NombreBarrio"].value_counts()


# In[24]:


df_result_poblacion_economico_idealista_inversiones_locales["NombreBarrio"].value_counts(normalize=True)


# In[25]:


plt.figure(figsize=(35,25))
fig = sns.countplot(x = 'NombreBarrio', data = df_result_poblacion_economico_idealista_inversiones_locales, 
                    palette=["#336699","#337799"])

plt.xlabel("Nombre Barrio")
plt.ylabel("Número de registros")
plt.title("Número de registros por Nombre de Barrio")
plt.xticks(rotation=90)
plt.show(fig)


# En este caso, se puede ver, que en casi todos los barrios se tiene el mismo número de registros menos en algunos casos. En estos barrios se puede dar la circustancia que el barrio sea de reciente creación o simplemente que no se dispone del dato para ese año.

# ## Variable cuantitativas o numéricas
# 
# A contnuación se realiza un estudio de las diferentes variables cuantitativas de las que disponemos. De todas ellas, se va a presentar una descripción con los estadisticos basicos, se representará su distribucion medianta un histograma y se calculará el valor de la Kurtosis y Skewness para interpretar la normalidad de la distribución. Por ultimo, se representara el gráfico boxplot y homologo grafico de violín.

# #### DatoPoblacion

# In[26]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacion = df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacion.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacion.describe()


# In[27]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoPoblacion", kde=True,  color="#336699")
plt.show()


# In[28]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacion'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacion'].kurt()}")


# In[29]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoPoblacion",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoPoblacion", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# ##### DatoPoblacionMenor16

# In[30]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMenor16 = df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMenor16.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMenor16.describe()


# In[31]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoPoblacionMenor16", kde=True,  color="#336699")
plt.show()


# In[32]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionMenor16'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionMenor16'].kurt()}")


# In[33]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoPoblacionMenor16",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoPoblacionMenor16", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoPoblacionEntre16y64

# In[34]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionEntre16y64 = df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionEntre16y64.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionEntre16y64.describe()


# In[35]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoPoblacionEntre16y64", kde=True,  color="#336699")
plt.show()


# In[36]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionEntre16y64'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionEntre16y64'].kurt()}")


# In[37]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoPoblacionEntre16y64",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoPoblacionEntre16y64", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoPoblacionMayor65

# In[38]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMayor65 = df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMayor65.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoPoblacionMayor65.describe()


# In[39]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoPoblacionMayor65", kde=True,  color="#336699")
plt.show()


# In[40]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionMayor65'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoPoblacionMayor65'].kurt()}")


# In[41]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoPoblacionMayor65",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoPoblacionMayor65", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoSobreEnvejecimiento

# In[42]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreEnvejecimiento = df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreEnvejecimiento.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreEnvejecimiento.describe()


# In[43]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoSobreEnvejecimiento", kde=True,  color="#336699")
plt.show()


# In[44]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreEnvejecimiento'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreEnvejecimiento'].kurt()}")


# In[45]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoSobreEnvejecimiento",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoSobreEnvejecimiento", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoSobreenvejecimientoHombres

# In[46]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoHombres = df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoHombres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoHombres.describe()


# In[47]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoSobreenvejecimientoHombres", kde=True,  color="#336699")
plt.show()


# In[48]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreenvejecimientoHombres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreenvejecimientoHombres'].kurt()}")


# In[49]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoSobreenvejecimientoHombres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoSobreenvejecimientoHombres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoSobreenvejecimientoMujeres

# In[50]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoMujeres = df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoMujeres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoSobreenvejecimientoMujeres.describe()


# In[51]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoSobreenvejecimientoMujeres", kde=True,  color="#336699")
plt.show()


# In[52]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreenvejecimientoMujeres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoSobreenvejecimientoMujeres'].kurt()}")


# In[53]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoSobreenvejecimientoMujeres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoSobreenvejecimientoMujeres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoFeminidad

# In[54]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoFeminidad = df_result_poblacion_economico_idealista_inversiones_locales.DatoFeminidad.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoFeminidad.describe()


# In[55]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoFeminidad", kde=True,  color="#336699")
plt.show()


# In[56]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoFeminidad'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoFeminidad'].kurt()}")


# In[57]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoFeminidad",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoFeminidad", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# ##### DatoParadosRegistrados

# In[58]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistrados = df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistrados.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistrados.describe()


# In[59]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoParadosRegistrados", kde=True,  color="#336699")
plt.show()


# In[60]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistrados'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistrados'].kurt()}")


# In[61]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoParadosRegistrados",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoParadosRegistrados", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoParadosRegistradosHombres

# In[62]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosHombres = df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosHombres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosHombres.describe()


# In[63]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoParadosRegistradosHombres", kde=True,  color="#336699")
plt.show()


# In[64]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistradosHombres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistradosHombres'].kurt()}")


# In[65]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoParadosRegistradosHombres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoParadosRegistradosHombres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoParadosRegistradosMujer

# In[66]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosMujer = df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosMujer.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoParadosRegistradosMujer.describe()


# In[67]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoParadosRegistradosMujer", kde=True,  color="#336699")
plt.show()


# In[68]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistradosMujer'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoParadosRegistradosMujer'].kurt()}")


# In[69]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoParadosRegistradosMujer",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoParadosRegistradosMujer", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesTrabajo

# In[70]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajo = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajo.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajo.describe()


# In[71]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesTrabajo", kde=True,  color="#336699")
plt.show()


# In[72]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajo'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajo'].kurt()}")


# In[73]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesTrabajo",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesTrabajo", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesTrabajoHombres

# In[74]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoHombres = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoHombres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoHombres.describe()


# In[75]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesTrabajoHombres", kde=True,  color="#336699")
plt.show()


# In[76]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajoHombres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajoHombres'].kurt()}")


# In[77]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesTrabajoHombres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesTrabajoHombres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesTrabajoMujer

# In[78]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoMujer = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoMujer.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesTrabajoMujer.describe()


# In[79]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesTrabajoMujer", kde=True,  color="#336699")
plt.show()


# In[80]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajoMujer'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesTrabajoMujer'].kurt()}")


# In[81]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesTrabajoMujer",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesTrabajoMujer", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesResidencia

# In[82]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidencia = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidencia.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidencia.describe()


# In[83]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesResidencia", kde=True,  color="#336699")
plt.show()


# In[84]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidencia'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidencia'].kurt()}")


# In[85]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesResidencia",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesResidencia", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesResidenciaHombres

# In[86]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaHombres = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaHombres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaHombres.describe()


# In[87]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesResidenciaHombres", kde=True,  color="#336699")
plt.show()


# In[88]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidenciaHombres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidenciaHombres'].kurt()}")


# In[89]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesResidenciaHombres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesResidenciaHombres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAfiliacionesResidenciaMujer

# In[90]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaMujer = df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaMujer.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAfiliacionesResidenciaMujer.describe()


# In[91]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAfiliacionesResidenciaMujer", kde=True,  color="#336699")
plt.show()


# In[92]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidenciaMujer'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAfiliacionesResidenciaMujer'].kurt()}")


# In[93]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAfiliacionesResidenciaMujer",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAfiliacionesResidenciaMujer", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAutonomo

# In[94]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomo = df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomo.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomo.describe()


# In[95]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAutonomo", kde=True,  color="#336699")
plt.show()


# In[96]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomo'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomo'].kurt()}")


# In[97]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAutonomo",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAutonomo", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAutonomosHombres

# In[98]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosHombres = df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosHombres.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosHombres.describe()


# In[99]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAutonomosHombres", kde=True,  color="#336699")
plt.show()


# In[100]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomosHombres'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomosHombres'].kurt()}")


# In[101]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAutonomosHombres",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAutonomosHombres", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### DatoAutonomosMujer

# In[102]:


df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosMujer = df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosMujer.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.DatoAutonomosMujer.describe()


# In[103]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="DatoAutonomosMujer", kde=True,  color="#336699")
plt.show()


# In[104]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomosMujer'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['DatoAutonomosMujer'].kurt()}")


# In[105]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "DatoAutonomosMujer",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "DatoAutonomosMujer", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### PrecioVentaEurosM2

# In[106]:


df_result_poblacion_economico_idealista_inversiones_locales.PrecioVentaEurosM2 = df_result_poblacion_economico_idealista_inversiones_locales.PrecioVentaEurosM2.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.PrecioVentaEurosM2.describe()


# In[107]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="PrecioVentaEurosM2", kde=True,  color="#336699")
plt.show()


# In[108]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['PrecioVentaEurosM2'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['PrecioVentaEurosM2'].kurt()}")


# In[109]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "PrecioVentaEurosM2",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "PrecioVentaEurosM2", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### PresupuestoGasto

# In[110]:


df_result_poblacion_economico_idealista_inversiones_locales.PresupuestoGasto = df_result_poblacion_economico_idealista_inversiones_locales.PresupuestoGasto.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.PresupuestoGasto.describe()


# In[111]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="PresupuestoGasto", kde=True,  color="#336699")
plt.show()


# In[112]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['PresupuestoGasto'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['PresupuestoGasto'].kurt()}")


# In[113]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "PresupuestoGasto",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "PresupuestoGasto", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### Gasto_Real

# In[114]:


df_result_poblacion_economico_idealista_inversiones_locales.Gasto_Real = df_result_poblacion_economico_idealista_inversiones_locales.Gasto_Real.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.Gasto_Real.describe()


# In[115]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="Gasto_Real", kde=True,  color="#336699")
plt.show()


# In[116]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['Gasto_Real'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['Gasto_Real'].kurt()}")


# In[117]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "Gasto_Real",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "Gasto_Real", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### NumeroLocales

# In[118]:


df_result_poblacion_economico_idealista_inversiones_locales.NumeroLocales = df_result_poblacion_economico_idealista_inversiones_locales.NumeroLocales.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.NumeroLocales.describe()


# In[119]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="NumeroLocales", kde=True,  color="#336699")
plt.show()


# In[120]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['NumeroLocales'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['NumeroLocales'].kurt()}")


# In[121]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "NumeroLocales",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "NumeroLocales", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# #### NumeroAlojamientos

# In[122]:


df_result_poblacion_economico_idealista_inversiones_locales.NumeroAlojamientos = df_result_poblacion_economico_idealista_inversiones_locales.NumeroAlojamientos.astype(float)

df_result_poblacion_economico_idealista_inversiones_locales.NumeroAlojamientos.describe()


# In[123]:


sns.histplot(data=df_result_poblacion_economico_idealista_inversiones_locales, x="NumeroAlojamientos", kde=True,  color="#336699")
plt.show()


# In[124]:


print(f"Skewness: {df_result_poblacion_economico_idealista_inversiones_locales['NumeroAlojamientos'].skew()}")
print(f"Kurtosis: {df_result_poblacion_economico_idealista_inversiones_locales['NumeroAlojamientos'].kurt()}")


# In[125]:


fig, axs = plt.subplots(ncols=2)

sns.boxplot(data=df_result_poblacion_economico_idealista_inversiones_locales, y = "NumeroAlojamientos",
            color = "#336699", ax=axs[0])

sns.violinplot(y = "NumeroAlojamientos", data = df_result_poblacion_economico_idealista_inversiones_locales, 
                color = "#336699", ax=axs[1]);


# In[126]:


df_result_poblacion_economico_idealista_inversiones_locales.skew()


# In[127]:


df_result_poblacion_economico_idealista_inversiones_locales.kurt()


# Con el estudio realizado se puede ver que existen multitud de 'outlayers" que deben ser estudiados. En el caso del dataset estudiado se ha visto que, pese a que los valores se consideran 'outlayers', son valores que se pueden considerar razonables. Es por esto que se decide dejar los valores ya que, precisamente, podrian ser de mucho valor a la hora de realizar el estudio que se quiere realizar.
# 
# Ademas, se analiza la asimetria de las variables mediante el estudio de la skewness que para datos grandes siempre arroja resultados similares a otros calculos.
# 
# El valor obtenido se puede interpretar de la siguiente manera:
# 
# - Si el valor de sesgo es menor que −1 o mayor que +1  la distribución será altamente sesgada.
# - si el valor de asimetría está entre −1 y +1 diremos que la distribución es moderadamente sesgada.
# 
# De la misma manera, se estudia el valor de la Kurtosis, o Curtosis, que se utiliza para describir la forma de la distribucion. Si la distribución tienen valores extremos, la forma será mas puntiaguda, por lo que se conoce a esta medida como forma de apuntalamiento. 
# 
# La interpretación de esta medida sera:
# 
# - Si Kurtosis < 3 la distribucion se considera Platicúrtica.
# - Si Kurtosis = 3 la distribucion se considera Mesocúrtica.
# - Si Kurtosis > 3 la distribucion se considera Leptocúrtica. 
# 
# Con este estudio lo que se pretende es conocer una caracteristica mas de la distribución. Una distribución muy asimnetrica podria indicar mucha desigualdad en los valores o que el esfuerzo a la hora de estudiar la variable se pueda centrar en unos pocos valores pero que aportan mucha información.
# 

# ### Estudio de correlaciones de las variables.
# 
# Por último, es muy interesante conocer la correlación que existe entre las diferentes varables del dataset. De esta manera se puede estudiar la influencia que puede tener una variable sobre otra.
# 
# Para realizar este estudio se utiliza el gráfico "pairplot" dentro de la libreria "seaborn"

# In[128]:


df_result_poblacion_economico_idealista_inversiones_locales.corr()


# In[129]:


sns.pairplot(df_result_poblacion_economico_idealista_inversiones_locales)

