import datetime
from functools import reduce
import math
from matplotlib import patches
from matplotlib.offsetbox import AnchoredText
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def get_date(x):
    print(x.values)
    return datetime.datetime(x['ano'],x['mes'],1)


def main():

    #En esta parte del código cargamos la data y la formateamos
    #Primero la data de cartera
    cartera = pd.read_csv("Datos/cartera.csv",sep= ',',encoding="ISO-8859-1")
    cartera.columns = cartera.columns.str.lower()
    cartera = cartera.rename(columns={'fecha': 'Fecha'})
    cartera['Fecha']=pd.to_datetime(cartera['Fecha'], format='%d/%m/%y')
    cartera['Fecha']=cartera['Fecha'].apply(lambda dt: dt.replace(day=1))

    cartera = cartera.pivot_table(index='Fecha', columns='sector_economico', values='deuda', aggfunc='sum')

    #Estos son los nombres de los distintos sectores economicos a los que se les presta
    cartera_vars=['A - AGRICULTURA, GANADERÍA, CAZA Y SILVICULTURA', 'B - PESCA',
        'C - EXPLOTACIÓN DE MINAS Y CANTERAS', 'D - INDUSTRIA MANUFACTURERA',
        'E - SUMINISTRO DE ELECTRICIDAD, GAS, VAPOR Y AIRE ACONDICIONADO',
        'F - CONSTRUCCIÓN',
        'G - COMERCIO AL POR MAYOR Y AL POR MENOR. REPARACIÓN DE LOS VEHÍCULOS DE MOTOR Y DE LAS MOTOCICLETAS',
        'H - ALOJAMIENTO Y SERVICIOS DE COMIDA',
        'I - TRANSPORTE Y ALMACENAMIENTO',
        'J - ACTIVIDADES FINANCIERAS Y DE SEGURO',
        'K - ACTIVIDADES INMOBILIARIAS, ALQUILER Y ACTIVIDADES EMPRESARIALES',
        'L - ADMINISTRACIÓN PÚBLICA Y DEFENSA: PLANES DE SEGURIDAD SOCIAL DE AFILIACIÓN OBLIGATORIA',
        'M - ENSEÑANZA',
        'N - SERVICIOS SOCIALES Y RELACIONADOS CON LA SALUD HUMANA',
        'P - ACTIVIDADES DE LOS HOGARES EN CALIDAD DE EMPLEADORES, ACTIVIDADES INDIFERENCIADAS DE PRODUCCIÓN DE BIENES Y SERVICIOS DE LOS HOGARES PARA USO PROPIO',
        'Q - ACTIVIDADES DE ORGANIZACIONES Y ÓRGANOS EXTRATERRITORIALES',
        'Y - CONSUMO DE BIENES Y SERVICIOS',
        'Z - COMPRA Y REMODELACIÓN DE VIVIENDAS']
    cartera=cartera.reset_index()
    cartera=cartera[['Fecha']+cartera_vars]

    #Aqui cargamos la data de impuestos
    impuestos = pd.read_csv("Datos/impuestos.csv")
    impuestos.columns = impuestos.columns.str.lower()
    impuestos['Fecha']=impuestos.apply(lambda x: get_date(x),axis=1)
    impuestos=impuestos[['Fecha','impuestos_totales']]

    #Aqui cargamos la data del IMAE  
    imae = pd.read_csv("Datos/imae.csv")
    imae.columns = imae.columns.str.lower()
    imae['Fecha']=imae.apply(lambda x: get_date(x),axis=1)
    imae=imae[['Fecha','imae']]
    


    #Seguimos cargando la data y formateándola
    #Aqui la data de remesas
    remesas = pd.read_csv("Datos/remesas.csv")
    remesas.columns = remesas.columns.str.lower()
    remesas['Fecha']=remesas.apply(lambda x: get_date(x),axis=1)
    remesas=remesas[['Fecha','remesas']]


    #Aqui la data de llegada de turistas
    turistas = pd.read_csv("Datos/turistas.csv")
    turistas.columns = turistas.columns.str.lower()
    turistas['Fecha']=turistas.apply(lambda x: get_date(x),axis=1)
    turistas=turistas[['Fecha','tur_mensual']]



    #Aqui cargamos la data de tasas pasivas
    tipm_interbancaria = pd.read_csv("Datos/tasas_pasivas.csv")
    tipm_interbancaria.columns = tipm_interbancaria.columns.str.lower()
    tipm_interbancaria['Fecha']=tipm_interbancaria.apply(lambda x: get_date(x),axis=1)
    tipm_interbancaria=tipm_interbancaria[['Fecha','tipm_interbancaria']]


    
    #Seguimos cargando la data y formateándola
    #Aqui la data del IPC
    IPC = pd.read_csv("Datos/ipc.csv")
    IPC.columns = IPC.columns.str.lower()
    IPC['Fecha']=IPC.apply(lambda x: get_date(x),axis=1)
    IPC=IPC[['Fecha','ipc']]



    #Aqui volvemos a cargar la data de tasas pasivas
    tipm_prom_p = pd.read_csv("Datos/tasas_pasivas.csv")
    tipm_prom_p.columns = tipm_prom_p.columns.str.lower()
    tipm_prom_p['Fecha']=tipm_prom_p.apply(lambda x: get_date(x),axis=1)
    tipm_prom_p=tipm_prom_p[['Fecha','tipm_prom_p']]


    #Aqui cargamos la data de tasas activas
    tiam_prom_p = pd.read_csv("Datos/tasas_activas.csv")
    tiam_prom_p.columns = tiam_prom_p.columns.str.lower()
    tiam_prom_p['Fecha']=tiam_prom_p.apply(lambda x: get_date(x),axis=1)
    tiam_prom_p=tiam_prom_p[['Fecha','tiam_prom_p']]


    #Aquí ponemos toda la data en un solo data frame
    #Aqui agrupamos los data frame generados anteriormente en una lista
    data_frames=[turistas,remesas,impuestos,tipm_interbancaria,tipm_prom_p,tiam_prom_p,cartera,IPC,imae]
    #Aqui juntamos toda la informacion de los data frame en uno solo alienando la fecha. La funcion reduce va uniendo los data frames iterativamente
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Fecha'],
                                                how='left'), data_frames)
    
    
    macro=df_merged.copy()
    macro.dropna()
    sc=StandardScaler()


    #En esta parte le hacemos transformaciones a la data
    df_merged=df_merged.sort_values(by=['Fecha'])
    df_merged['tipm_interbancaria']=df_merged['tipm_interbancaria']-df_merged['ipc'].pct_change(12)*100

    df_merged['IMAE_original']=df_merged['imae'].astype(float)
    df_merged['Fecha']=pd.to_datetime(df_merged['Fecha'])
    df_merged['mes']=df_merged['Fecha'].dt.month

    df_merged[['impuestos_totales','remesas']+cartera_vars]=df_merged[['impuestos_totales','remesas']+cartera_vars].shift(2)

    
    df_merged[['impuestos_totales','tur_mensual','remesas']]=(df_merged[['impuestos_totales','tur_mensual','remesas']]).diff(3)


    df_merged[cartera_vars]=((df_merged[cartera_vars])).diff(3)
    df_merged['imae_cambio']=(df_merged[['imae']]).diff(12)
    df_merged['imae_cambio_target']=(df_merged['imae_cambio']).shift(-2)
    df_merged['imae_cambio_11']=(df_merged[['imae']]).diff(10)

    df_merged['imae_shifted_11']=df_merged['imae'].shift(10)

    #Aquí creamos objetos vacios para guardar resultados
    imae_cambio=[]
    predicciones=[]
    predicciones_inferior=[]
    predicciones_superior=[]
    Fechas=[]
    sc=StandardScaler()
    imae_shifted_11=[]
    explained_variance={}

    #En esta parte hacemos el ejercicio de pseudo-out-of-sample, por eso la iteracion
    #En esta parte del codigo vamos estimando el modelo y generando predicciones expandiendo la muestra iterativamente
    for i in range(1,len(df_merged)):

        #En esta parte vamos armando la muestra para el modelo para cada iteracion
        df_train=df_merged.iloc[:i]
        df_train=df_train.dropna()
        #Solo estimamos el modelo con un minimo de 30 observaciones
        if(len(df_train)<30):
            continue
        df_test=df_merged.iloc[i-10:i+1]
        df_test=df_test.dropna(subset=cartera_vars+['impuestos_totales','tur_mensual','remesas']+['tipm_interbancaria','tiam_prom_p','tipm_prom_p'])

        if(len(df_test)==0):
            continue


        #Aquí tomamos las features predictivas por grupo (Tasas, Macro y Cartera)
        X_tasas_train,X_tasas_test=df_train[['tipm_interbancaria','tiam_prom_p','tipm_prom_p']],df_test[['tipm_interbancaria','tiam_prom_p','tipm_prom_p']]
        X_macro_train,X_macro_test=df_train[['impuestos_totales','tur_mensual','remesas']],df_test[['impuestos_totales','tur_mensual','remesas']]
        X_cartera_train,X_cartera_test=df_train[cartera_vars],df_test[cartera_vars]


        #En esta parte estandarizamos la data para extraer los factores posteriormente
        X_tasas_train = sc.fit_transform(X_tasas_train)
        X_tasas_test = sc.transform(X_tasas_test)

        X_macro_train = sc.fit_transform(X_macro_train)
        X_macro_test = sc.transform(X_macro_test)

        X_cartera_train = sc.fit_transform(X_cartera_train)
        X_cartera_test = sc.transform(X_cartera_test)

        #Aquí inicializamos el PCA para las variables macro
        pca = PCA(n_components = 1)

        #Calculamos el PCA para las variables Macro y la varianza explicada por el mismo
        X_macro_train = pca.fit_transform(X_macro_train)
        explained_variance['macro']=pca.explained_variance_ratio_
        X_macro_test = pca.transform(X_macro_test)
        macro_df_train=pd.DataFrame(X_macro_train)
        macro_df_test=pd.DataFrame(X_macro_test)

        #Seguimos con el PCA, ahora para las variables de cartera
        pca = PCA(n_components = 2)

        #Calculamos los PCA para las variables de Cartera y calculamos la varianza explicada
        X_cartera_train = pca.fit_transform(X_cartera_train)
        explained_variance['cartera']=pca.explained_variance_ratio_

        X_cartera_test = pca.transform(X_cartera_test)
        cartera_df_train=pd.DataFrame(X_cartera_train)
        cartera_df_test=pd.DataFrame(X_cartera_test)

        pca = PCA(n_components = 1)

        #Lo mismo con las tasas
        X_tasas_train = pca.fit_transform(X_tasas_train)
        explained_variance['tasas']=pca.explained_variance_ratio_

        X_tasas_test = pca.transform(X_tasas_test)
        tasas_df_train=pd.DataFrame(X_tasas_train,columns=['F'])
        tasas_df_test=pd.DataFrame(X_tasas_test,columns=['F'])


        #Aquí creamos seasonal dummies. Las mismas son variables que toman el valor de 1 cuando uno se encuentra en un mes en particular y 0 para el resto de meses. 
        #Son un total de 12, una por mes
        seasonal_train=pd.DataFrame(df_train['mes'].values,columns=['mes'])
        seasonal_test=pd.DataFrame(df_test['mes'].values,columns=['mes'])

        for i2 in range(1,13):
            seasonal_train['month_dummy'+str(i2)]=(seasonal_train['mes']==i2).astype(int)
            seasonal_test['month_dummy'+str(i2)]=(seasonal_test['mes']==i2).astype(int)
        seasonal_train=seasonal_train.drop(columns=['mes'])
        seasonal_test=seasonal_test.drop(columns=['mes'])

        #Aquí calculamos la diferencia de orden 11 del IMAE. La misma será utilizada como feature predictiva.
        imae_shifted_train=df_train[['imae_cambio_11']].reset_index()
        imae_shifted_test=df_test[['imae_cambio_11']].reset_index()

        #Luego de calculados los PCA, unimos todas las features predictivas construidas
        df_train_iter = reduce(lambda  left,right: pd.merge(left,right,how='left',left_index=True, right_index=True), [cartera_df_train,macro_df_train,tasas_df_train,imae_shifted_train])
        df_test_iter = reduce(lambda  left,right: pd.merge(left,right,how='left',left_index=True, right_index=True), [cartera_df_test,macro_df_test,tasas_df_test,imae_shifted_test])
        
        #Constante para la forecasting regression  
        df_train_iter['constant']=1
        df_test_iter['constant']=1

        df_train_iter['Fecha']=df_train['Fecha'].values
        #Aquí eliminamos las observaciones de la pandemia
        df_train_iter=df_train_iter.loc[(df_train_iter['Fecha']<datetime.datetime(2020,2,1))|(df_train_iter['Fecha']>datetime.datetime(2022,1,1))]
        df_train_iter=df_train_iter.drop(columns=['Fecha'])

        #Creamos un objeto para la variable target, o sea, la variable a predecir
        y_train=df_train[['imae_cambio_target']]
        y_train['Fecha']=df_train['Fecha'].values

        #Aquí eliminamos las observaciones de la pandemia
        y_train=y_train.loc[(y_train['Fecha']<datetime.datetime(2020,2,1))|(y_train['Fecha']>datetime.datetime(2022,1,1))]
        y_train=y_train.drop(columns=['Fecha'])

        #Aquí alineamos el target con las features predictivas
        df_train_iter=df_train_iter.reset_index(drop=True).dropna()
        y_train=y_train.reset_index().loc[list(df_train_iter.index)]
        y_train=y_train['imae_cambio_target'].values

        x_train=df_train_iter.values
        x_test=df_test_iter.iloc[-1,:].values

        #Aquí estimamos la forecasting regression
        model = sm.OLS(y_train, x_train).fit()
        prediction_interval = model.get_prediction(x_test).conf_int(0.90)
        prediction = model.predict(x_test)

        print(df_merged.iloc[i]['Fecha'])

        #En esta parte vamos guardando los resultados
        Fechas.append(df_merged.iloc[i]['Fecha'])
        predicciones.append(prediction[-1])
        imae_cambio.append(df_merged.iloc[i]['imae_cambio'])
        predicciones_superior.append(prediction_interval[0][1])
        predicciones_inferior.append(prediction_interval[0][0])
        imae_shifted_11.append(df_merged.iloc[i]['imae_shifted_11'])

    results=pd.DataFrame(Fechas,columns=['Fecha'])

    print(explained_variance)

    #Aquí vamos organizando los resultados guardados
    results['imae_shifted_11']=imae_shifted_11
    results['imae_cambio_real']=imae_cambio
    results['prediccion_inferior']=predicciones_inferior
    results['prediccion']=predicciones

    results['Fecha']=results['Fecha'].shift(-2)
    results['imae_cambio_real']=results['imae_cambio_real'].shift(-2)
    results['imae_shifted_11']=results['imae_shifted_11']
    results['Fecha'].iloc[-1]=results.iloc[-3]['Fecha']+(results.iloc[-4]['Fecha']-results.iloc[-6]['Fecha'])
    results['Fecha'].iloc[-2]=results.iloc[-3]['Fecha']+(results.iloc[-4]['Fecha']-results.iloc[-5]['Fecha'])



    results['prediccion_superior']=predicciones_superior
    results['prediccion_superior_diferencia']=(results['prediccion_superior'])
    results['prediccion_superior_nivel']=(results['prediccion_superior'])+results['imae_shifted_11']

    results['imae_cambio_real_diferencia']=results['imae_cambio_real']
    results['imae_cambio_real_nivel']=(results['imae_cambio_real'])+results['imae_shifted_11']

    results['prediccion_inferior_diferencia']=(results['prediccion_inferior'])
    results['prediccion_inferior_nivel']=(results['prediccion_inferior'])+results['imae_shifted_11']
    results['prediccion_diferencia']=(results['prediccion'])
    results['prediccion_nivel']=(results['prediccion'])+results['imae_shifted_11']

    results=results.loc[(results['Fecha']<datetime.datetime(2020,2,1))|(results['Fecha']>datetime.datetime(2022,1,1))]
    

    #En esta parte vamos ploteando la serie de tiempo de los resultados para la diferencia interanual
    sns.lineplot(data=results,x='Fecha',y='imae_cambio_real_diferencia',label="IMAE",color='blue')
    sns.lineplot(data=results,x='Fecha',y='prediccion_diferencia',label="Predicciones",color='black')
    sns.lineplot(data=results,x='Fecha',y='prediccion_superior_diferencia',label="Límite superior (nivel de confianza del 90%)",linestyle='dashed',color='gray')
    sns.lineplot(data=results,x='Fecha',y='prediccion_inferior_diferencia',label="Límite inferior (nivel de confianza del 90%)",linestyle='dashed',color='gray')
    plt.title('Predicción del Indicador Mensual de Actividad Económica (IMAE) t+2 - Diferencia')

    x_start = datetime.datetime(2020,2,1)
    x_end = datetime.datetime(2022,1,1)
    ax = plt.gca()
    ax.axvspan(x_start, x_end, color='black')
    ax.set_ylim(np.min(results.iloc[10:]['prediccion_inferior_diferencia']*.8), np.max(results.iloc[10:]['prediccion_superior_diferencia'])*1.2)    # Set y-axis limits

    plt.grid(True)

    plt.show()

    results['residuo']=results['imae_cambio_real_diferencia']-results['prediccion_diferencia']
    

    data=results['residuo'].dropna().values
    x_start = datetime.datetime(2020,2,1)
    x_end = datetime.datetime(2022,1,1)

    #En esta parte vamos ploteando la serie de tiempo de los resultados para el nivel
    ax = plt.gca()
    ax.axvspan(x_start, x_end, color='gray')
    sns.lineplot(data=results,x='Fecha',y='imae_cambio_real_nivel',label="IMAE",color='blue')
    sns.lineplot(data=results,x='Fecha',y='prediccion_nivel',label="Predicciones",color='black')
    sns.lineplot(data=results,x='Fecha',y='prediccion_superior_nivel',label="Límite superior (nivel de confianza del 90%)",linestyle='dashed',color='gray')
    sns.lineplot(data=results,x='Fecha',y='prediccion_inferior_nivel',label="Límite inferior (nivel de confianza del 90%)",linestyle='dashed',color='gray')
    plt.title('Predicción del Indicador Mensual de Actividad Económica (IMAE) t+2 - Nivel')
    plt.grid(True)
    plt.show()

    mean = np.mean(data)
    second_moment = np.mean((data - mean)**2)
    third_moment = np.mean((data - mean)**3) / np.std(data)**3  
    fourth_moment = np.mean((data - mean)**4) / np.std(data)**4  
    print(f"Promedio (primer momento): {mean}")
    print(f"Varianza (segundo momento): {second_moment}")
    print(f"Asimetría (tercer momento): {third_moment}")
    print(f"Curtosis (cuarto moment): {fourth_moment}")

    #Aqui ploteamos el histograma de los errores de predicción
    g=sns.displot(data=results['residuo'],label="Residuos",kde=True, palette={'A': 'blue', 'B': 'red'},)
    stats_text = (
        f'Promedio: {mean:.2f}\n'
        f'Varianza: {second_moment:.2f}\n'
        f'Desviación estándar: {math.sqrt(second_moment):.2f}\n'

        f'Asimetría: {third_moment:.2f}\n'
        f'Curtosis: {fourth_moment:.2f}\n'
    )
    ax = g.ax  #Acceso a el objeto de ejes desde FacetGrid 
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(x=xlim[1], y=ylim[1], # Coordenadas para el texto
            s=stats_text,     # El texto a desplegar
            ha='right',               # Alineamiento horizontal ('right' alinea el texto hacia la coordenada x)
            va='top',                 # Alineamiento vertical ('top' alinea el texto hacia la coordenada y)
            fontsize=10,              # Tamano de texto
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Color del background para mas facil lectura 

    plt.grid(True)
    plt.title('Residuos de la diferencia t+2')
    plt.show()


    results['prediccion']=results['residuo']+results['prediccion_diferencia'].dropna().iloc[-1]
    data=results['prediccion'].dropna().values

    mean = np.mean(data)
    second_moment = np.mean((data - mean)**2)
    third_moment = np.mean((data - mean)**3) / np.std(data)**3  
    fourth_moment = np.mean((data - mean)**4) / np.std(data)**4  
    print(f"Promedio (primer momento): {mean}")
    print(f"Varianza (segundo momento): {second_moment}")
    print(f"Asimetría (tercer momento): {third_moment}")
    print(f"Curtosis (cuarto moment): {fourth_moment}")
    g=sns.displot(data=results['prediccion'],label="Prediccion",kde=True, palette={'A': 'blue', 'B': 'red'},)

    #Aquí ploteamos el density forecast para t+2
    stats_text = (
        f'Promedio: {mean:.2f}\n'
        f'Varianza: {second_moment:.2f}\n'
        f'Desviación estándar: {math.sqrt(second_moment):.2f}\n'

        f'Asimetría: {third_moment:.2f}\n'
        f'Curtosis: {fourth_moment:.2f}\n'
    )
    ax = g.ax  #Acceso a el objeto de ejes desde FacetGrid
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(x=xlim[1], y=ylim[1],  # Coordenadas para el texto
            s=stats_text,     # El texto a desplegar
            ha='right',               # Alineamiento horizontal ('right' alinea el texto hacia la coordenada x)
            va='top',                 # Alineamiento vertical ('top' alinea el texto hacia la coordenada y)
            fontsize=10,              # Tamano de texto
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Color del background para mas facil lectura 

    plt.title('Density Forecast t+2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()