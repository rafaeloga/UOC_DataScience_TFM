#!/usr/bin/env python
# coding: utf-8

# # Instalación de paquetes python.
# 
# Se instalan los paquetes necesarios para realizar web scraping de la página de Idealista.

# In[ ]:


get_ipython().system('pip3 install beautifulsoup4 --upgrade')
get_ipython().system('pip3 install requests')
get_ipython().system('pip3 install selenium')
get_ipython().system('pip3 install webdriver-manager')


# # Importación de las librerias necesarias.
# 
# Se importan las librerias necesarias para la ejecución del código

# In[1]:


import requests
import bs4 as bs
import pandas as pd
import numpy as np
import os
import requests


# # Web Scraping.
# 
# Se extraen desde el navegador Chrome las cookies y las headers que se utilizan para que el portal no detecte el script como un robot. 

# In[49]:


cookies = {
    'atuserid': '%7B%22name%22%3A%22atuserid%22%2C%22val%22%3A%2265923206-622f-451c-9784-07de945faea3%22%2C%22options%22%3A%7B%22end%22%3A%222024-04-23T09%3A39%3A10.029Z%22%2C%22path%22%3A%22%2F%22%7D%7D',
    'atidvisitor': '%7B%22name%22%3A%22atidvisitor%22%2C%22val%22%3A%7B%22vrn%22%3A%22-582065-%22%7D%2C%22options%22%3A%7B%22path%22%3A%22%2F%22%2C%22session%22%3A15724800%2C%22end%22%3A15724800%7D%7D',
    'didomi_token': 'eyJ1c2VyX2lkIjoiMTg3MGRkNjItMTIwOC02YzAyLWFiOTctMmI0NjdhMjE0OTU1IiwiY3JlYXRlZCI6IjIwMjMtMDMtMjNUMTA6NTQ6NTcuNjk5WiIsInVwZGF0ZWQiOiIyMDIzLTAzLTIzVDEwOjU0OjU3LjY5OVoiLCJ2ZXJzaW9uIjoyLCJwdXJwb3NlcyI6eyJlbmFibGVkIjpbImFuYWx5dGljcy1IcEJKcnJLNyIsImdlb2xvY2F0aW9uX2RhdGEiLCJkZXZpY2VfY2hhcmFjdGVyaXN0aWNzIl19LCJ2ZW5kb3JzIjp7ImVuYWJsZWQiOlsiZ29vZ2xlIiwiYzpsaW5rZWRpbi1tYXJrZXRpbmctc29sdXRpb25zIiwiYzptaXhwYW5lbCIsImM6YWJ0YXN0eS1MTGtFQ0NqOCIsImM6aG90amFyIiwiYzp5YW5kZXhtZXRyaWNzIiwiYzpiZWFtZXItSDd0cjdIaXgiLCJjOmFwcHNmbHllci1HVVZQTHBZWSIsImM6dGVhbGl1bWNvLURWRENkOFpQIiwiYzp0aWt0b2stS1pBVVFMWjkiLCJjOmlkZWFsaXN0YS1MenRCZXFFMyIsImM6aWRlYWxpc3RhLWZlUkVqZTJjIl19LCJhYyI6IkFGbUFDQUZrLkFGbUFDQUZrIn0=',
    'euconsent-v2': 'CPpE-EAPpE-EAAHABBENC8CsAP_AAE7AAAAAF5wBQAIAAtAC2AKQBeYAAACA0AGAAIJTEoAMAAQSmKQAYAAglMQgAwABBKYdABgACCUwSADAAEEphkAGAAIJTCoAMAAQSmAA.f_gACdgAAAAA',
    '_gcl_au': '1.1.1276745818.1679568907',
    '_hjFirstSeen': '1',
    '_hjSession_250321': 'eyJpZCI6ImNlNzJlZTNiLTcwZjgtNGUwMC1iYTY3LTZiYzJlYWRiZGI1NiIsImNyZWF0ZWQiOjE2Nzk1Njg5MDczMDMsImluU2FtcGxlIjp0cnVlfQ==',
    '_hjAbsoluteSessionInProgress': '1',
    '_fbp': 'fb.1.1679568907315.1686135134',
    '_tt_enable_cookie': '1',
    '_ttp': 'yM4YuSR3x8nNbGap7-IQwv0N6UQ',
    '_hjSessionUser_250321': 'eyJpZCI6ImRiOGY1ZTk2LTc2NzAtNTRjZi05OTI0LTE5YWQxNTVjNTYxMyIsImNyZWF0ZWQiOjE2Nzk1Njg5MDcyOTYsImV4aXN0aW5nIjp0cnVlfQ==',
    'utag_main': 'v_id:01870dd621c40055fa87fd3b86b405065008605d00bd0$_sn:2$_se:52$_ss:0$_st:1679572064088$ses_id:1679568895718%3Bexp-session$_pn:52%3Bexp-session$_prevVtSource:directTraffic%3Bexp-1679572495936$_prevVtCampaignCode:%3Bexp-1679572495936$_prevVtDomainReferrer:idealista.com%3Bexp-1679572495936$_prevVtSubdomaninReferrer:www.idealista.com%3Bexp-1679572495936$_prevVtUrlReferrer:https%3A%2F%2Fwww.idealista.com%2F%3Bexp-1679572495936$_prevVtCampaignLinkName:%3Bexp-1679572495936$_prevVtCampaignName:%3Bexp-1679572495936$_prevVtRecommendationId:%3Bexp-1679572495936$_prevCompletePageName:255%20%3E%20%3Bexp-1679573864198$_prevLevel2:255%3Bexp-1679573864198$_prevCompleteClickName:',
    '_hjIncludedInSessionSample_250321': '1',
    '_hjHasCachedUserAttributes': 'true',
    'datadome': '2P7dWk72DzQo67jt6ZnKkP~BTcw5Xo69lBPxmMdXnRh71OiYYv3iTlOQPGsXj0hDa~_KrDehoeumQZXMjDz_pGWW4qo6fkc0RD2Q2yqxI9zVgd7hNTm6CQADZQk1cs7E',
    'outbrain_cid_fetch': 'true',
}

headers = {
    'authority': 'www.idealista.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'es-ES,es;q=0.9,en;q=0.8,ca;q=0.7',
    'cache-control': 'max-age=0',
    # 'cookie': 'atuserid=%7B%22name%22%3A%22atuserid%22%2C%22val%22%3A%2265923206-622f-451c-9784-07de945faea3%22%2C%22options%22%3A%7B%22end%22%3A%222024-04-23T09%3A39%3A10.029Z%22%2C%22path%22%3A%22%2F%22%7D%7D; atidvisitor=%7B%22name%22%3A%22atidvisitor%22%2C%22val%22%3A%7B%22vrn%22%3A%22-582065-%22%7D%2C%22options%22%3A%7B%22path%22%3A%22%2F%22%2C%22session%22%3A15724800%2C%22end%22%3A15724800%7D%7D; didomi_token=eyJ1c2VyX2lkIjoiMTg3MGRkNjItMTIwOC02YzAyLWFiOTctMmI0NjdhMjE0OTU1IiwiY3JlYXRlZCI6IjIwMjMtMDMtMjNUMTA6NTQ6NTcuNjk5WiIsInVwZGF0ZWQiOiIyMDIzLTAzLTIzVDEwOjU0OjU3LjY5OVoiLCJ2ZXJzaW9uIjoyLCJwdXJwb3NlcyI6eyJlbmFibGVkIjpbImFuYWx5dGljcy1IcEJKcnJLNyIsImdlb2xvY2F0aW9uX2RhdGEiLCJkZXZpY2VfY2hhcmFjdGVyaXN0aWNzIl19LCJ2ZW5kb3JzIjp7ImVuYWJsZWQiOlsiZ29vZ2xlIiwiYzpsaW5rZWRpbi1tYXJrZXRpbmctc29sdXRpb25zIiwiYzptaXhwYW5lbCIsImM6YWJ0YXN0eS1MTGtFQ0NqOCIsImM6aG90amFyIiwiYzp5YW5kZXhtZXRyaWNzIiwiYzpiZWFtZXItSDd0cjdIaXgiLCJjOmFwcHNmbHllci1HVVZQTHBZWSIsImM6dGVhbGl1bWNvLURWRENkOFpQIiwiYzp0aWt0b2stS1pBVVFMWjkiLCJjOmlkZWFsaXN0YS1MenRCZXFFMyIsImM6aWRlYWxpc3RhLWZlUkVqZTJjIl19LCJhYyI6IkFGbUFDQUZrLkFGbUFDQUZrIn0=; euconsent-v2=CPpE-EAPpE-EAAHABBENC8CsAP_AAE7AAAAAF5wBQAIAAtAC2AKQBeYAAACA0AGAAIJTEoAMAAQSmKQAYAAglMQgAwABBKYdABgACCUwSADAAEEphkAGAAIJTCoAMAAQSmAA.f_gACdgAAAAA; _gcl_au=1.1.1276745818.1679568907; _hjFirstSeen=1; _hjSession_250321=eyJpZCI6ImNlNzJlZTNiLTcwZjgtNGUwMC1iYTY3LTZiYzJlYWRiZGI1NiIsImNyZWF0ZWQiOjE2Nzk1Njg5MDczMDMsImluU2FtcGxlIjp0cnVlfQ==; _hjAbsoluteSessionInProgress=1; _fbp=fb.1.1679568907315.1686135134; _tt_enable_cookie=1; _ttp=yM4YuSR3x8nNbGap7-IQwv0N6UQ; _hjSessionUser_250321=eyJpZCI6ImRiOGY1ZTk2LTc2NzAtNTRjZi05OTI0LTE5YWQxNTVjNTYxMyIsImNyZWF0ZWQiOjE2Nzk1Njg5MDcyOTYsImV4aXN0aW5nIjp0cnVlfQ==; utag_main=v_id:01870dd621c40055fa87fd3b86b405065008605d00bd0$_sn:2$_se:52$_ss:0$_st:1679572064088$ses_id:1679568895718%3Bexp-session$_pn:52%3Bexp-session$_prevVtSource:directTraffic%3Bexp-1679572495936$_prevVtCampaignCode:%3Bexp-1679572495936$_prevVtDomainReferrer:idealista.com%3Bexp-1679572495936$_prevVtSubdomaninReferrer:www.idealista.com%3Bexp-1679572495936$_prevVtUrlReferrer:https%3A%2F%2Fwww.idealista.com%2F%3Bexp-1679572495936$_prevVtCampaignLinkName:%3Bexp-1679572495936$_prevVtCampaignName:%3Bexp-1679572495936$_prevVtRecommendationId:%3Bexp-1679572495936$_prevCompletePageName:255%20%3E%20%3Bexp-1679573864198$_prevLevel2:255%3Bexp-1679573864198$_prevCompleteClickName:; _hjIncludedInSessionSample_250321=1; _hjHasCachedUserAttributes=true; datadome=2P7dWk72DzQo67jt6ZnKkP~BTcw5Xo69lBPxmMdXnRh71OiYYv3iTlOQPGsXj0hDa~_KrDehoeumQZXMjDz_pGWW4qo6fkc0RD2Q2yqxI9zVgd7hNTm6CQADZQk1cs7E; outbrain_cid_fetch=true',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
}


# Cargamos el dataset de los barrios para poder obtener los diferentes distritos que vamos a consultar en Idealista.

# In[2]:


#Obtenemos los distritos:
barrios_df = pd.read_csv('datos/raw/Barrios.csv', decimal=',', delimiter=';',header=0)[['COD_BAR', 'NOMBRE', 'CODDIS','NOMDIS']]                                                                                        
distritos = barrios_df.sort_values('NOMDIS')[['CODDIS','NOMDIS']].drop_duplicates()
distritos


# # Obtención de precios de venta y alquiler.
# 
# Para cada distrito de los obtenidos anteriormente, se realiza la extracción tanto de los precios de alquiler como de venta desde la página de Idealista. 

# In[ ]:


#Obtenemos precio de venta y alquiler
for tipo in ['venta','alquiler']:
        for distrito in distritos:
            print('Obteniendo datos de {} del distrito: {}...'.format(tipo,distrito.NOMDIS))
            url = 'https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/{}/madrid-comunidad/madrid-provincia/madrid/{}/historico/'.format(tipo,distrito.lower().replace(' ','-'))
            response = requests.get(url, cookies=cookies, headers=headers)
            df = pd.read_html(response.text)[0]
            df = df.loc[df['precio_{}_m2'.format(tipo)] != 'n.d.']
            df_final.to_csv('datos/raw/precio{}/{}.csv'.format(tipo), index=false)


# Para cada csv se tratan los datos de venta de los pisos. Se recorren todos los csv de los distritos independientes, se da el formato correspondiente y se almacena el csv resultate con todos los datos historicos de venta en único csv.

# In[4]:


#Tratamos los datos de precio de venta:
df_total = pd.DataFrame()
for index, distrito in distritos.iterrows():
    coddis = distrito['CODDIS']
    nomdis = distrito['NOMDIS']
    file = 'datos/raw/precioventa/{}.csv'.format(nomdis.lower().split('-')[0].strip().replace(' ','-'))
    print(file)
    df = pd.read_csv(file, delimiter=';',header=0)
    df['CODDIS'] = coddis
    df['NOMDIS'] = nomdis
    df[['Mes','ANIO']] = df.Mes.str.split(' ', expand=True)
    df = df[df['Mes']=='Diciembre']
    df['Precio m2'] = df['Precio m2'].str.replace(' €/m2','')
    df = df[['ANIO', 'CODDIS','NOMDIS','Precio m2']]
    df.rename(columns = {'Precio m2':'PrecioVentaEurosM2'}, inplace = True)
    df_total = pd.concat([df_total,df])

df_total.to_csv('datos/raw/precioventahistorico.csv', sep=';',index=False)


# Se realiza el mismo proceso que se ha realizado para los precios de venta, pero en este caso para los precios de alquiler. Se genera un unico fichero con el historico de precios de alquiler de todos los distritos.

# In[5]:


#Tratamos los datos de precio de alquiler:
df_total = pd.DataFrame()
for index, distrito in distritos.iterrows():
    coddis = distrito['CODDIS']
    nomdis = distrito['NOMDIS']
    file = 'datos/raw/precioalquiler/{}.csv'.format(nomdis.lower().split('-')[0].strip().replace(' ','-'))
    print(file)
    df = pd.read_csv(file, delimiter=';',header=0)
    df['CODDIS'] = coddis
    df['NOMDIS'] = nomdis
    df[['Mes','ANIO']] = df.Mes.str.split(' ', expand=True)
    df = df[df['Mes']=='Diciembre']
    df['Precio m2'] = df['Precio m2'].str.replace(' €/m2','')
    df = df[['ANIO', 'CODDIS','NOMDIS','Precio m2']]
    df.rename(columns = {'Precio m2':'PrecioAlquilerEurosM2'}, inplace = True)
    df_total = pd.concat([df_total,df])

df_total.to_csv('datos/raw/precioalquilerhistorico.csv', sep=';',index=False)

