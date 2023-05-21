#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing, cluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# ## Carga y tratamiento de Datos
# 

# Lo primera vamos a cargar el dataset con los datos a utilizar.
# 

# In[2]:


df = pd.read_csv('datos/silver/result_datos_poblacion_economicos_idealista_inversiones_locales.csv', delimiter=';',header=0)

df_2010_2022 = df[df['Anio'].isin([2012,2022])]
df_2010_2022=df_2010_2022.fillna(0)
list(df_2010_2022.columns)


# In[3]:


df_features =['CodigoBarrio','NombreBarrio',
 'DatoPoblacion',
 'DatoPoblacionMenor16',
 'DatoPoblacionEntre16y64',
 'DatoPoblacionMayor65',
 'DatoSobreEnvejecimiento',
 'DatoSobreenvejecimientoHombres',
 'DatoSobreenvejecimientoMujeres',
 'DatoFeminidad',
 'DatoNacionalidadExtranjera',
 'DatoParadosRegistrados',
 'DatoParadosRegistradosHombres',
 'DatoParadosRegistradosMujer',
 'DatoAfiliacionesTrabajo',
 'DatoAfiliacionesTrabajoHombres',
 'DatoAfiliacionesTrabajoMujer',
 'DatoAfiliacionesResidencia',
 'DatoAfiliacionesResidenciaHombres',
 'DatoAfiliacionesResidenciaMujer',
 'DatoAutonomo',
 'DatoAutonomosHombres',
 'DatoAutonomosMujer',
 'PrecioVentaEurosM2',
 'PrecioAlquilerEurosM2',
 'PresupuestoGasto',
 'Gasto_Real',
 'NumeroLocales',
 'NumeroAlojamientos']


# In[4]:




df_2010_2022 = df_2010_2022.groupby(['CodigoBarrio','NombreBarrio']).agg(
            DatoPoblacion=('DatoPoblacion',min),
            DatPoblacionMenor16=('DatoPoblacionMenor16',min),
            DatPoblacionEntre16y64=('DatoPoblacionEntre16y64',min),
            DatoPoblacionMayor65=('DatoPoblacionMayor65',min),
            DatoSobreEnvejecimiento=('DatoSobreEnvejecimiento',min),
            DatoSobreenvejecimientoHombres=('DatoSobreenvejecimientoHombres',min),
            DatoSobreenvejecimientoMujeres=('DatoSobreenvejecimientoMujeres',min),
            DatoFeminidad=('DatoFeminidad',min),
            DatoNacionalidadExtranjera=('DatoNacionalidadExtranjera',min),
            DatoParadosRegistrados=('DatoParadosRegistrados',min),
            DatoParadosRegistradosHombres=('DatoParadosRegistradosHombres',min),
            DatoParadosRegistradosMujer=('DatoParadosRegistradosMujer',min),
            DatoAfiliacionesTrabajo=('DatoAfiliacionesTrabajo',min),
            DatoAfiliacionesTrabajoHombres=('DatoAfiliacionesTrabajoHombres',min),
            DatoAfiliacionesTrabajoMujer=('DatoAfiliacionesTrabajoMujer',min),
            DatoAfiliacionesResidencia=('DatoAfiliacionesResidencia',min),
            DatoAfiliacionesResidenciaHombres=('DatoAfiliacionesResidenciaHombres',min),
            DatoAfiliacionesResidenciaMujer=('DatoAfiliacionesResidenciaMujer',min),
            DatoAutonomo=('DatoAutonomo',min),
            DatoAutonomosHombres=('DatoAutonomosHombres',min),
            DatoAutonomosMujer=('DatoAutonomosMujer',min),
            PrecioVentaEurosM2=('PrecioVentaEurosM2',min),
            PrecioAlquilerEurosM2=('PrecioAlquilerEurosM2',min),
            PresupuestoGasto=('PresupuestoGasto',min),
            Gasto_Real=('Gasto_Real',min),
            NumeroLocales=('NumeroLocales',min),
            NumeroAlojamientos=('NumeroAlojamientos',min)
).reset_index() 

#df_2010_2022.columns = pd.MultiIndex.from_tuples(list(zip(["features"]*29 ,df_features)))

df_2010_2022.describe()


# In[5]:


df_2010_2022.isna().sum()


# In[6]:


mask = np.zeros_like(df_2010_2022.corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (12,12))
sns.heatmap(df_2010_2022.corr(), 
            annot=False,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmin = -1.0,
            vmax = 1.0,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()


# In[7]:


df_norm = df_2010_2022.copy()
df_norm = df_norm.drop(columns=['NombreBarrio', 'CodigoBarrio'])
x = df_norm.values  


scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)


df_norm = pd.DataFrame(x_scaled)

df_norm


# In[8]:


# Elbow method
sse = []
ks = range(2, 10)
for k in ks:
        k_means_model = cluster.KMeans(n_clusters=k, random_state=55)
        k_means_model.fit(df_norm)
        sse.append(k_means_model.inertia_)

fig, axis = plt.subplots(figsize=(9, 6))
axis.set_title('Método del codo para una k óptima')
axis.set_xlabel('k')
axis.set_ylabel('SSE')
plt.plot(ks, sse, marker='o')
plt.tight_layout()
plt.show()


# In[9]:


# Silhouette method

ks = range(3, 10)
sils = []
for k in ks:
        clusterer = KMeans(n_clusters=k, random_state=55)
        cluster_labels = clusterer.fit_predict(df_norm)
        silhouette_avg = silhouette_score(df_norm, cluster_labels)
        sils.append(silhouette_avg)
        print("Para n_clusters =", k, "La media para el método de la silueta:",
              silhouette_avg)

fig, axis = plt.subplots(figsize=(9, 6))
axis.set_title('Método de la silueta')
axis.set_xlabel('k')
axis.set_ylabel('Silhouette')
plt.plot(ks, sils, marker='o')
plt.tight_layout()
plt.show()


# In[10]:


# Método Mutual Informaks = range(3, 10))

# Inicializar una lista para almacenar los valores de información mutua
mutual_infos = []
ks = range(3, 10)
for k in ks:

    # Ejecutar el algoritmo k-means
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(df_norm)
    
    # Calcular la información mutua entre las etiquetas de los clusters y las etiquetas verdaderas (no utilizadas en este ejemplo)
    mi = mutual_info_score(labels, labels)
    mutual_infos.append(mi)

# Encontrar el valor máximo de información mutua y el correspondiente número de clusters óptimo
optimal_k = k_values[np.argmax(mutual_infos)]
print("Número óptimo de clusters:", optimal_k)

fig, axis = plt.subplots(figsize=(9, 6))
axis.set_title('Método Mutual Information')
axis.set_xlabel('k')
axis.set_ylabel('Mutual Information')
plt.plot(ks, mutual_infos, marker='o')
plt.tight_layout()
plt.show()


# Es nuestro caso parece que el valor optimo de k estará entre 5 y 6. 

# In[ ]:


cluster = KMeans(n_clusters=8, random_state=55)
cluster_labels = cluster.fit_predict(df_norm)

df_result = df_2010_2022[["CodigoBarrio",'NombreBarrio']]
df_result['K_means_k8'] = cluster_labels
    

cluster = KMeans(n_clusters=9, random_state=10)
cluster_labels = cluster.fit_predict(df_norm)

df_result['K_means_k9'] = cluster_labels

df_result.to_csv('datos/gold/result_k8_k9.csv', index=False)

df_result


# In[ ]:


#Gaussian Model

from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.mixture import GaussianMixture

gaussian_model = GaussianMixture(n_components=8)
cluster_labels = gaussian_model.fit_predict(df_norm)

df_result['Gaussian_8'] = cluster_labels


gaussian_model = GaussianMixture(n_components=9)
cluster_labels = gaussian_model.fit_predict(df_norm)

df_result['Gaussian_9'] = cluster_labels

df_result.to_csv('datos/gold/result_k8_k9_g8_g9.csv', index=False)

cluster_labels


# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.cluster import Birch


birch_model = Birch(threshold=0.03, n_clusters=8)
cluster_labels = birch_model.fit_predict(df_norm)
df_result['Birch_8'] = cluster_labels

birch_model = Birch(threshold=0.03, n_clusters=9)
cluster_labels = birch_model.fit_predict(df_norm)
df_result['Birch_9'] = cluster_labels

df_result.to_csv('datos/gold/result_k8_k9_g8_g9_b8_b9.csv', index=False)

cluster_labels


# In[ ]:


from sklearn.cluster import AffinityPropagation


affinity_model = AffinityPropagation(damping=0.7)
cluster_labels = affinity_model.fit_predict(df_norm)

df_result['Affinity'] = cluster_labels

df_result.to_csv('datos/gold/result_k5_k6_g5_g6_b5_b6_aff.csv', index=False)

cluster_labels


# In[ ]:


from sklearn.cluster import MeanShift

mean_model = MeanShift()
cluster_labels = mean_model.fit_predict(df_norm)

df_result['MeanShift'] = cluster_labels

df_result.to_csv('datos/gold/result_k5_k6_g5_g6_b5_b6_aff_mean.csv', index=False)

cluster_labels


# In[ ]:


from sklearn.cluster import OPTICS

optics_model = OPTICS(eps=0.75, min_samples=5)
cluster_labels = optics_model.fit_predict(df_norm)

df_result['Optics'] = cluster_labels

df_result.to_csv('datos/gold/result_k5_k6_g5_g6_b5_b6_aff_mean_optics.csv', index=False)

cluster_labels


# In[ ]:


from sklearn.cluster import AgglomerativeClustering


agglomerative_model = AgglomerativeClustering(n_clusters=8)

cluster_labels = agglomerative_model.fit_predict(df_norm)

df_result['Agg_8'] = cluster_labels



agglomerative_model = AgglomerativeClustering(n_clusters=9)

cluster_labels = agglomerative_model.fit_predict(df_norm)

df_result['Agg_9'] = cluster_labels

df_result.to_csv('datos/gold/result_k8_k9_g8_g9_b8_b9_aff_mean_optics_ag8_ag9.csv', index=False)

cluster_labels

