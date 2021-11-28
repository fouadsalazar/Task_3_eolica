import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import plotly.graph_objects as go
import math
import statistics
from windrose import WindroseAxes
from windrose import WindAxes
import matplotlib.pyplot as plt
import scipy

def leer(año):
    datos = pd.read_csv('{}.txt'.format(año), sep=",")
    
    # Se arregla el formato de las fechas para poder trabajarlo
    datos["DATE (MM/DD/YYYY)"]= pd.to_datetime(datos["DATE (MM/DD/YYYY)"])
    datos['Año'] = datos["DATE (MM/DD/YYYY)"].dt.year
    datos['Mes'] = datos["DATE (MM/DD/YYYY)"].dt.month
    datos['Dia'] = datos["DATE (MM/DD/YYYY)"].dt.day
    
    # Se arregla el formato de las horas 
    hora=[]
    for i in range(len(datos)):
        hora.append(datetime.strptime(datos["MST"][i],"%H:%M"))
    datos['Hora'] = hora
    datos['Hora'] = datos["Hora"].dt.hour
    
    return(datos)

    # Limpieza de datos
def corregir_datos(datos):
    columna=["Avg Wind Speed @ 2m [m/s]","Avg Wind Speed @ 5m [m/s]",
             "Avg Wind Speed @ 10m [m/s]","Avg Wind Speed @ 20m [m/s]","Avg Wind Speed @ 50m [m/s]",
             "Avg Wind Speed @ 80m [m/s]"]

    datos.fillna(0,inplace=True)
    
    for i in range(len(datos)):
        for j in range(len(columna)):
            if i > 0  and i < (len(datos)-1):
                if datos[columna[j]][i] < 0:
                    datos[columna[j]][i] = (datos[columna[j]][i-1] + datos[columna[j]][i+1])/2
    return (datos)
        
def parametros_weibull(año,direccion,alturas,direc):
    for i in (alturas):
        windspeed = direccion[f'Avg Wind Speed @ {i}m [m/s]']
        ax = WindAxes.from_ax()
        ax, params = ax.pdf(windspeed, bins=100)
        ax.set(title=f'Direccion: {direc.upper()} - Vel a {i} metros, {año}',
              xlabel ='Velocidad [m/s]', ylabel ='Frecuencia')
        return (params)
    
def weibull_general(año,datos,alturas):
    for i in (alturas):
        windspeed = datos[f'Avg Wind Speed @ {i}m [m/s]']
        ax = WindAxes.from_ax()
        ax, params = ax.pdf(windspeed, bins=100)
        ax.set(title=f'Representación de la distribución anual de velocidades del viento a {i} metros para datos del año {año}',
              xlabel ='Velocidad [m/s]', ylabel ='Frecuencia')
        return (params)
        
def direcciones(datos,altura):
    norte1=datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]<11.25)]
    norte2=datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>348.75)]
    n = pd.concat([norte1, norte2])
    n.index = range(n.shape[0])
    
    nne = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>11.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<33.75)]
    nee = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>33.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<56.25)]
    ene = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>56.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<78.75)]
    e = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>78.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<101.25)]
    ese = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>101.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<123.75)]
    se = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>123.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<146.25)]
    sse = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>146.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<168.75)]
    s = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>168.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<191.25)]
    ssw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>191.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<213.75)]
    sw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>213.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<236.25)]
    wsw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>236.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<258.75)]
    w = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>258.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<281.25)]
    wnw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>281.25)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<303.75)]
    nw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>303.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<326.25)]
    nnw = datos.loc[(datos[f"Avg Wind Direction @ {altura}m [deg]"]>326.75)&(datos[f"Avg Wind Direction @ {altura}m [deg]"]<348.75)]
    
    return(n,nne,nee,ene,e,ese,se,sse,s,ssw,sw,wsw,w,wnw,nw,nnw)

def frecuencias(wd,ws,ax):
    ax.bar(wd, ws, normed=True, nsector=16)
    table = ax._info['table']

    direction = ax._info['dir']
    wd_freq = np.sum(table, axis=0)
    plt.bar(np.arange(16), wd_freq, align='center')
    xlabels = ('N','','N-E','','E','','S-E','','S','','S-O','','O','','N-O','')
    xticks=np.arange(16)
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xlabels)
    plt.title("Frecuencia en % de la velocidad del viento para las distintas direcciones")
    plt.ylabel("Frecuencia [%]")
    plt.show()
    return (wd_freq)

def analisis_weibull_completo(año,alturas,datos,ax):
    n,nne,nee,ene,e,ese,se,sse,s,ssw,sw,wsw,w,wnw,nw,nnw = direcciones(datos,"80")
    sector = ["n","nne","nee","ene","e","ese","se","sse","s","ssw","sw","wsw","w","wnw","nw","nnw"]
    forma,escala,velocidad=[],[],[]
    x=0
    for i in (n,nne,nee,ene,e,ese,se,sse,s,ssw,sw,wsw,w,wnw,nw,nnw):
        vel = i['Avg Wind Speed @ 80m [m/s]'].mean()
        params=parametros_weibull(año,i,alturas,sector[x])
        forma.append(params[1])
        escala.append(params[3])
        velocidad.append(vel)
        x+=1
        
    ws = datos[f'Avg Wind Speed @ 80m [m/s]']
    wd = datos[f'Avg Wind Direction @ 80m [deg]']
    wd_freq = frecuencias(wd,ws,ax)
    respuesta = {"Sector":sector,"Frecuencias":wd_freq,"Factor de forma":forma,"Factor de escala":escala,
                 "Velocidad prom":velocidad}
    respuesta1 = pd.DataFrame(respuesta)
    respuesta1.to_csv(f"respuesta1_{año}.csv")
    
ax = WindroseAxes.from_ax()

for i in (2017,2018,2019):
    datos = corregir_datos(leer(i))
    analisis_weibull_completo(i,["80"],datos,ax)
    weibull_general(i,datos,["80"])
    
for i in (2017,2018,2019):
    datos = corregir_datos(leer(i))
    
    for j in (2,4,6,8,10):
        if j==5:
            datos = datos.loc[(datos["Avg Wind Speed @ 80m [m/s]"]<j)]
        else:
            datos = datos.loc[(datos["Avg Wind Speed @ 80m [m/s]"]>(j-2))&((datos["Avg Wind Speed @ 80m [m/s]"]<(j)))]
            print("Analisis del año",i,"y el rango de velocidades",j-2,"a",j)
            print()
            analisis_weibull_completo(i,["80"],datos,ax)
            weibull_general(i,datos,["80"])
