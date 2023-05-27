# UOC_DataScience_TFM 

Repositorio con el código relativo al proyecto fin de master de Ciencia de Datos de la UOC.  

  

En este repositorio se puede encontrar la versión final del código empleado. Así como las diferentes fuentes de datos que se han utilizado en la realización del proyecto. 

  

## Código del proyecto 

Dentro del código podemos encontrar distintos notebooks y su correspondiente código exportado a python. Entre ellos podemos encontrar: 

  

  - **descargaPortalEstadistico.ipynb** que incluye el código empleado para descargar los datos provenientes del portal estadístico. Se incluye la correspondiente versión en python llamada **descargaPortalEstadistico.py**. 

  - **descargaIdealista.ipynb** que incluye el código empleado para descargar los datos de precios de compra y alquiler de viviendas. Se incluye la correspondiente versión en python llamada **descargaIdealista.py**. 

  - **descargaCensoLocales.ipynb** con el código utilizado para descargar el cesno de locales y de alojamientos de cada barrio. Su correspondiente versión en python se llama **descargaCensoLocales.py** 

  - **mergeDataFrames.ipynb** que incluye el código por el que se unen los diferentes dataframes en uno completo. Se incluye la correspondiente versión en python llamada **mergeDataFrames.py**. 

  - **exploratoryDataAnalysis.ipynb** que incluye el análisis exploratorio completo de todas las variables del dataset resultado del merge de todos los dataset parciales. Se incluye su correspondiente versión en python **exploratoryDataAnalysis.py**. 

  - **modelosnosupervisados.ipynb** incluye la implementación de los diferentes modelos de clustering que se han utilizado en el desarrollo del proyecto. Su correspondiente versión en python es **modelosnosupervisados.py**. 

  

  

## Carpetas de datos. 

  

En esta carpeta se pueden encontrar tres directorios diferentes: 

  

- **Raw**: contiene los datos en bruto que se han obtenido de diferentes fuentes. Se pueden encontrar los datos descargados del portal estadístico o de Idealista. 

- **Silver**: Contiene ficheros que han sido tratados y mergeados de la carpeta raw. El fichero result_datos_poblacion_economicos_idealista_inversiones_locales.csv contiene el set completo de datos.  

- **Gold**: Contiene ficheros de datos que han sido enriquecidos o modificados para el tratamiento de datos por los modelos supervisados o son los ficheros definitivos que se han subido a CARTO para su posterior representación en el mapa. 

  

## Evolución de Indicadores. 

  

En esta carpeta se muestran los mapas generados con CARTO. En ellos se pueden observar, para cada indicador, la evolución que han tenido a lo largo de los años de una manera rápida. 
