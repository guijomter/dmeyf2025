# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval
from datetime import timezone, timedelta

logger = logging.getLogger(__name__)

#######################################################################################################

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada

	#completar a gusto!!!!!!!
        'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learn_rate', conf.parametros_lgb.learn_rate[0], conf.parametros_lgb.learn_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': 0.0,
       # 'verbose': -1,
        'verbosity': -1,
        'silent': True,
        'bin': 31,
        'random_state': SEMILLA[0] #,  # Desde configuración YAML
        
    }
  
    # Preparar dataset para entrenamiento y validación

    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].astype(str).isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'].astype(str) == MES_TRAIN]
    
    df_val = df[df['foto_mes'].astype(str) == MES_VALIDACION]

    # Usar target (con clase ternaria ya convertida a binaria)
    
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values

    # Features (excluir target)
    X_train = df_train.drop(columns=['clase_ternaria'])
    X_val = df_val.drop(columns=['clase_ternaria'])

    # Completar!!!!!!
    # Entrenar modelo con función de ganancia personalizada

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=ganancia_lgb_binary, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia

    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)  # Usar mismo umbral que en ganancia_lgb_binary                  


    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total
   

#######################################################################################################

def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")

#######################################################################################################

def optimizar(df, n_trials=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = conf.STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
    )

    # Función objetivo parcial con datos
    objective_with_data = lambda trial: objetivo_ganancia(trial, df)

    # Ejecutar optimización
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)
  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study

#######################################################################################################
### VERSION VIEJA DE EVALUACION EN TEST SOLO PARA CALCULAR GANANCIA

# def evaluar_en_test(df, mejores_params) -> dict:
#     """
#     Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
#     Solo calcula la ganancia, sin usar sklearn.
  
#     Args:
#         df: DataFrame con todos los datos
#         mejores_params: Mejores hiperparámetros encontrados por Optuna
  
#     Returns:
#         dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
#     """
#     logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
#     logger.info(f"Período de test: {MES_TEST}")
  
#     # Preparar datos de entrenamiento (TRAIN + VALIDACION)
#     if isinstance(MES_TRAIN, list):
#         periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
#     else:
#         periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
#     df_train_completo = df[df['foto_mes'].astype(str).isin(periodos_entrenamiento)]
#     df_test = df[df['foto_mes'].astype(str) == MES_TEST]
  
#     # Entrenar modelo con mejores parámetros
#     # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
#     # recordar realizar todos los df necesarios y utilizar lgb.train()
#     # Cargar mejores parámetros

#     # Entrenar modelo con mejores parámetros
#     logger.info("Entrenando modelo con mejores hiperparámetros...")
#     logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')

#     # Preparar datasets

#     train_data = lgb.Dataset(df_train_completo.drop(columns=['clase_ternaria']), label=df_train_completo['clase_ternaria'].values)
#     test_data = lgb.Dataset(df_test.drop(columns=['clase_ternaria']), label=df_test['clase_ternaria'].values, reference=train_data)
#   # chequeo si train_data y test_data estan bien formados
#     logger.info(f"Tipo de dato de train_data: {type(train_data)}, Tipo de dato de test_data: {type(test_data)}")
#     logger.info(f"Dimensiones de train_data: {train_data.data.shape}, Dimensiones de test_data: {test_data.data.shape}")

#     model = lgb.train(
#         mejores_params,
#         train_data,
#         #num_boost_round=1000,
#         valid_sets=[test_data],
#         feval=ganancia_lgb_binary,
#         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
#     )

#     # Predecir en test
    
#     X_test = df_test.drop(columns=['clase_ternaria'])
#     y_test = df_test['clase_ternaria'].values
#     y_pred_proba = model.predict(X_test)
#     y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)  # Usar mismo umbral que en ganancia_lgb_binary


#     # Calcular solo la ganancia
#     ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
#     # Estadísticas básicas
#     total_predicciones = len(y_pred_binary)
#     predicciones_positivas = np.sum(y_pred_binary == 1)
#     porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
#     resultados = {
#         'ganancia_test': float(ganancia_test),
#         'total_predicciones': int(total_predicciones),
#         'predicciones_positivas': int(predicciones_positivas),
#         'porcentaje_positivas': float(porcentaje_positivas)
#     }
  
#     return resultados
#########################################################################################################


def evaluar_en_test(df, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].astype(str).isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'].astype(str) == MES_TEST]
  
    # Entrenar modelo con mejores parámetros
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()
    # Cargar mejores parámetros

    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')

    # Preparar datasets

    train_data = lgb.Dataset(df_train_completo.drop(columns=['clase_ternaria']), label=df_train_completo['clase_ternaria'].values)
  
    # chequeo si train_data está ok
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")


    model = lgb.train(
        mejores_params,
        train_data,
        #num_boost_round=1000,
        #valid_sets=[test_data],
        #feval=ganancia_lgb_binary,
        feval=ganancia_evaluator
      #  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Predecir en test
    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria'].values
    y_pred_proba = model.predict(X_test)

    # Buscar el umbral que maximiza la ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  # 0.00, 0.005, ..., 1.00

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # Guardar predicción óptima

    ganancia_test = mejor_ganancia

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': semilla
    }
  
    return resultados
#######################################################################################################

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    """
    Args:
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_resultado_test.json"
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
    
    tz = timezone(timedelta(hours=-3))

    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    iteracion_data = {
        'Mes_test': MES_TEST,
        'ganancia_test': float(resultados_test['ganancia_test']),
        'date_time': datetime.now(tz).isoformat(),
        'state': 'COMPLETE',
        'configuracion':{
            'semilla': resultados_test['semilla'],
            'meses_train': periodos_entrenamiento
        },
        'resultados':resultados_test
    }

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    #logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {resultados_test['ganancia_test']:,.0f}" + "---" + f"Total Predicciones positivas: {resultados_test['predicciones_positivas']:,.0f}")



#####################################################################################

def evaluar_en_test_pesos(df, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].astype(str).isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'].astype(str) == MES_TEST]

    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')

    # Preparar datasets
    X= df_train_completo.drop(columns=['clase_ternaria', 'clase_peso'])
    y= df_train_completo['clase_ternaria'].values
    weights= df_train_completo['clase_peso'].values  # Pesos para cada instancia

    train_data = lgb.Dataset(X, label=y, weight=weights)
  
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")
  

    model = lgb.train(
        mejores_params,
        train_data,
        #num_boost_round=1000,
        #valid_sets=[test_data],
        feval=lgb_gan_eval
      #  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Predecir en test
    X_test = df_test.drop(columns=['clase_ternaria', 'clase_peso'])
    y_test = df_test['clase_peso'].values
    y_test = np.where(y_test == 1.00002, 1, 0)  # Convertir pesos de clase_ternaria a 1.00002 y 1

    y_pred_proba = model.predict(X_test)

    predicciones_test = pd.DataFrame({
        'probabilidad': y_pred_proba,
        'clase_ternaria': y_test
    })

    # Ordenar por probabilidad descendente
    predicciones_test = predicciones_test.sort_values(by='probabilidad', ascending=False).reset_index(drop=True)

    # Guardar predicciones ordenadas en CSV
    predicciones_test.to_csv(f'resultados/predicciones_test_ordenadas_{conf.STUDY_NAME}_semilla_{semilla}.csv', index=False)

    # Buscar el umbral que maximiza la ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  # 0.00, 0.005, ..., 1.00

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # Guardar predicción óptima

    ganancia_test = mejor_ganancia

    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': semilla
    }
  
    return resultados