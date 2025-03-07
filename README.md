Este es el repositorio de la blogpost series “Prediciendo el IMAE en un data-rich environment”

### Detalle

En este repositorio tenemos los archivos que nos permiten ejecutar una proyección del IMAE (indicador de actividad económica para la República Dominicana) cuando contamos con un conjunto amplio de información. Específicamente proyectamos a uno, dos y tres meses. Para esta proyección con toda esta data utilizamos para modelizar un modelo de factores, una de las herramientas de Big Data por excelencia en macroeconomía. El modelo es evaluado usando un procedimiento típico de pseudo-out-of-sample como es usual al predecir series de tiempo. 


### Configurar

Instalar los paquetes necesarios de python para ejecutar los scripts de la carpeta `src`

```bash
pip install -r requirements.txt
```

### Ejecutar la predicción

Para ejecutar las predicciones se debe ejectar cada script de la carpeta `src`, por ejemplo:

```bash
python t+1.py
```

El resultado debe ser un conjunto de gráficos que se automuestran y un DataFrame con los resultados.

### Scripts

Se encuentran en la carpeta `src`:

- **t+1.py**: predice a 1 mes el IMAE
- **t+2.py**: predice a 2 meses el IMAE
- **t+3.py**: predice a 3 meses el IMAE