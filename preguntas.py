"""
Pronostico de la resistencia del concreto usando redes neuronales
-----------------------------------------------------------------------------------------

La descripción del problema está disponible en:

https://jdvelasq.github.io/courses/notebooks/sklearn_supervised_10_neural_networks/1-02_pronostico_de_la_resistencia_del_concreto.html

"""

import pandas as pd


def pregunta_01():
    """
    Carga y separación de los datos en `X` `y`
    """
    # Lea el archivo `concrete.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("concrete.csv", sep=",")  

    # Asigne la columna `strength` a la variable `y`.
    y = df["strength"]  

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()  

    # Remueva la columna `strength` del DataFrame `X`.
    X.pop("strength")  

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación del dataset.
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos de ejemplo y asigne los resultados a `X` y `y`.
    x, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12453. Use el 75% de los patrones para entrenamiento.
    (  
        x_train,  
        x_test,  
        y_train,  
        y_test,  
    ) = train_test_split(  
        x,  
        y,  
        test_size=0.25,  
        random_state=12453,  
    )  

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return x_train, x_test, y_train, y_test


def pregunta_03():
    """
    Construcción del pipeline
    """

    # Importe MLPRegressor
    # Importe MinMaxScaler
    # Importe Pipeline
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline

    # Cree un pipeline que contenga un estimador MinMaxScaler y un estimador
    # MLPRegressor
    pipeline = Pipeline(
        steps=[
            (
                "minmaxscaler",
                MinMaxScaler(),  
            ),
            (
                "mlpregressor",
                MLPRegressor(),  
            ),
        ],
    )

    # Retorne el pipeline
    return pipeline


def pregunta_04():
    """
    Creación de la malla de búsqueda
    """

    # Importe GridSearchCV
    from sklearn.model_selection import GridSearchCV

    # Cree una malla de búsqueda para el objecto GridSearchCV
    # con los siguientes parámetros de búesqueda:
    #   * De 1 a 8 neuronas en la capa oculta
    #   * Activación con la función `relu`.
    #   * Tasa de aprendizaje adaptativa
    #   * Momentun con valores de 0.7, 0.8 y 0.9
    #   * Tasa de aprendijzaje inicial de 0.01, 0.05, 0.1
    #   * Un máximo de 5000 iteraciones
    #   * Use parada temprana

    param_grid = {
        'mlpregressor__hidden_layer_sizes': [(i,) for i in range(1, 9)],  
        'mlpregressor__activation': ['relu'],  
        'mlpregressor__learning_rate': ['adaptive'],  
        'mlpregressor__momentum': [0.7, 0.8, 0.9],  
        'mlpregressor__learning_rate_init': [0.01, 0.05, 0.1],  
        'mlpregressor__max_iter': [5000],  
        'mlpregressor__early_stopping': [True],  
    }

    estimator = pregunta_03()

    # Especifique un objeto GridSearchCV con el pipeline y la malla de búsqueda,
    # y los siguientes parámetros adicionales:
    #  * Validación cruzada con 5 particiones
    #  * Compare modelos usando r^2
    gridsearchcv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv = 5,  
        scoring= 'r2',  
    )

    return gridsearchcv


def pregunta_05():
    """
    Evalue el modelo obtenido.
    """

    # Importe mean_squared_error
    from sklearn.metrics import mean_squared_error

    # Cargue las variables.
    x_train, x_test, y_train, y_test = pregunta_02()

    # Obtenga el objeto GridSearchCV
    estimator = pregunta_04()

    # Entrene el estimador
    estimator.fit(x_train, y_train)  #

    # Pronostique para las muestras de entrenamiento y validacion
    y_trian_pred = estimator.predict(x_train)  
    y_test_pred = estimator.predict(x_test)  

    # Calcule el error cuadrático medio de las muestras
    mse_train = mean_squared_error(  
        y_train,  
        y_trian_pred,  
    )
    mse_test = mean_squared_error(  
        y_test,  
        y_test_pred,  
    )

    # Retorne el mse de entrenamiento y prueba
    return mse_train, mse_test
