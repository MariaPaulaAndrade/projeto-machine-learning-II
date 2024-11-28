import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from requests.adapters import HTTPAdapter
from requests.sessions import Session

PARAM_GRID_RF = {
    'n_estimators': [10, 50, 100],
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, None]
}

PARAM_GRID_GB = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [10, 20, 30]
}

PARAM_GRID_KNN = {'n_neighbors': [3, 5, 7, 9]}

PARAM_GRID_SVM = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

PARAM_GRID_XGB = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.6, 0.8, 1.0]
}

PARAM_GRID_ADABOOST = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1, 10],
}

PARAM_GRID_GRADIENTBOOST = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}

PARAM_GRID_KMEANS = {
    'n_clusters': [2, 3, 5, 10],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [300, 500]
}

PARAM_GRID_DBSCAN = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [5, 10, 15],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

PARAM_GRID_HIERARCHICAL = {
    'n_clusters': [2, 3, 5, 10],
    'linkage': ['ward', 'complete', 'average', 'single'],
    'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
}

def model_train_test(df: pd.DataFrame, target: str, test_size=0.3, random_state=42) -> list:
    """
    Divide os dados em conjuntos de treino e teste para modelagem.

    Esta função separa um DataFrame em variáveis preditoras (X) e a variável 
    alvo (y), e então realiza a divisão desses dados em conjuntos de treino 
    e teste usando o método `train_test_split` do Scikit-learn.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame contendo os dados de entrada.
    target : str
        O nome da coluna que representa a variável alvo (y).
    test_size : float, opcional, padrão=0.3
        A proporção do conjunto de dados que será usado como teste. 
        Deve ser um valor entre 0 e 1.
    random_state : int, opcional, padrão=42
        Define o estado aleatório para garantir reprodutibilidade nos
        resultados da divisão.

    Retorna:
    --------
    list
        Uma lista contendo quatro elementos:
        - X_train: pd.DataFrame
          Conjunto de treino das variáveis preditoras.
        - X_test: pd.DataFrame
          Conjunto de teste das variáveis preditoras.
        - y_train: pd.Series
          Conjunto de treino da variável alvo.
        - y_test: pd.Series
          Conjunto de teste da variável alvo.
    """

    X, y = df.drop(columns=target), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def model_pre_process(df: pd.DataFrame, target: str, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    """
    Preprocessa os dados de treino e teste para modelagem.

    Esta função aplica transformações nos dados categóricos e numéricos, 
    incluindo codificação one-hot para variáveis categóricas e escalonamento 
    robusto para variáveis numéricas. Os dados transformados são retornados 
    prontos para serem usados em modelos de machine learning.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame original contendo os dados.
    target : str
        O nome da coluna que representa a variável alvo.
    X_train : pd.DataFrame
        Conjunto de treino das variáveis preditoras.
    X_test : pd.DataFrame
        Conjunto de teste das variáveis preditoras.

    Retorna:
    --------
    list
        Uma lista contendo dois elementos:
        - X_train_processed: np.ndarray
          Conjunto de treino preprocessado.
        - X_test_processed: np.ndarray
          Conjunto de teste preprocessado.
    """

    df = df.drop(columns=target).copy()

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numerical_transformer = Pipeline(
        steps = [
            ('scaler', RobustScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


def mlflow_up(experiment: str, url='http://127.0.0.1:5000') -> None:
    """
    Configura o ambiente para rastreamento de experimentos com MLflow.

    Esta função define a URI de rastreamento do MLflow, configura o nome do experimento 
    e habilita a autologging para modelos Scikit-learn, permitindo o registro automático 
    de métricas, parâmetros e artefatos.

    Parâmetros:
    -----------
    experiment : str
        O nome do experimento a ser configurado no MLflow.
    url : str, opcional, padrão='http://127.0.0.1:5000'
        A URI de rastreamento do servidor MLflow.

    Retorna:
    --------
    None
        Esta função não retorna nenhum valor.
    """

    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog(silent=True)

def supervised_rand_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Realiza uma busca aleatória de hiperparâmetros em um modelo supervisionado com registro no MLflow.

    Esta função utiliza `RandomizedSearchCV` para encontrar os melhores hiperparâmetros para o modelo 
    fornecido, com base nos dados de treino e validação cruzada. Durante o processo, métricas e 
    parâmetros são registrados no MLflow para rastreamento de experimentos.

    Parâmetros:
    -----------
    model : object
        O modelo de machine learning a ser ajustado.
    param_grid : dict
        O espaço de busca para os hiperparâmetros.
    X_train : array-like ou pd.DataFrame
        Dados de treino das variáveis preditoras.
    X_test : array-like ou pd.DataFrame
        Dados de teste das variáveis preditoras.
    y_train : array-like ou pd.Series
        Variável alvo para treino.
    y_test : array-like ou pd.Series
        Variável alvo para teste.
    cv : int, opcional, padrão=5
        Número de divisões para validação cruzada.
    n_jobs : int, opcional, padrão=-1
        Número de threads para execução paralela. Use -1 para utilizar todos os processadores disponíveis.
    verbose : int, opcional, padrão=1
        Nível de detalhamento do log durante a execução da busca.

    Retorna:
    --------
    list
        Uma lista contendo:
        - run_id: str
          O ID da execução registrada no MLflow.
        - best_model: object
          O melhor estimador obtido pela busca aleatória.
    """

    with mlflow.start_run(run_name=f'Supervised_RandomSearchCV_{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        rand_search = RandomizedSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )

        rand_search.fit(X_train, y_train)
        
        best_model = rand_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(rand_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {rand_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def supervised_grid_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Realiza uma busca exaustiva de hiperparâmetros em um modelo supervisionado com registro no MLflow.

    Esta função utiliza `GridSearchCV` para testar todas as combinações possíveis de 
    hiperparâmetros fornecidas no `param_grid`, com base nos dados de treino e validação cruzada. 
    Durante o processo, métricas e parâmetros são registrados no MLflow para rastreamento de experimentos.

    Parâmetros:
    -----------
    model : object
        O modelo de machine learning a ser ajustado.
    param_grid : dict
        O espaço de busca contendo todas as combinações possíveis de hiperparâmetros.
    X_train : array-like ou pd.DataFrame
        Dados de treino das variáveis preditoras.
    X_test : array-like ou pd.DataFrame
        Dados de teste das variáveis preditoras.
    y_train : array-like ou pd.Series
        Variável alvo para treino.
    y_test : array-like ou pd.Series
        Variável alvo para teste.
    cv : int, opcional, padrão=5
        Número de divisões para validação cruzada.
    n_jobs : int, opcional, padrão=-1
        Número de threads para execução paralela. Use -1 para utilizar todos os processadores disponíveis.
    verbose : int, opcional, padrão=1
        Nível de detalhamento do log durante a execução da busca.

    Retorna:
    --------
    list
        Uma lista contendo:
        - run_id: str
          O ID da execução registrada no MLflow.
        - best_model: object
          O melhor estimador obtido pela busca exaustiva.
    """

    with mlflow.start_run(run_name=f'Supervised_GridSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(grid_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {grid_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def supervised_bayesian_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Configura o ambiente para rastreamento de experimentos com MLflow.

    Esta função define a URI de rastreamento do MLflow, configura o nome do experimento 
    e habilita a autologging para modelos Scikit-learn, permitindo o registro automático 
    de métricas, parâmetros e artefatos.

    Parâmetros:
    -----------
    experiment : str
        O nome do experimento a ser configurado no MLflow.
    url : str, opcional, padrão='http://127.0.0.1:5000'
        A URI de rastreamento do servidor MLflow.

    Retorna:
    --------
    None
        Esta função não retorna nenhum valor.
    """

    with mlflow.start_run(run_name=f'Supervised_BayesSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        bayesian_search = BayesSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        bayesian_search.fit(X_train, y_train)
        
        best_model = bayesian_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(bayesian_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {bayesian_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def unsupervised_rand_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Realiza uma busca aleatória de hiperparâmetros em um modelo não supervisionado com registro no MLflow.

    Esta função utiliza `RandomizedSearchCV` para encontrar os melhores hiperparâmetros para o modelo 
    fornecido, com base nos dados de treino e validação cruzada. Durante o processo, os parâmetros 
    otimizados são registrados no MLflow para rastreamento de experimentos.

    Parâmetros:
    -----------
    model : object
        O modelo de aprendizado não supervisionado a ser ajustado.
    param_grid : dict
        O espaço de busca para os hiperparâmetros.
    X_train : array-like ou pd.DataFrame
        Dados de treino das variáveis preditoras.
    cv : int, opcional, padrão=5
        Número de divisões para validação cruzada.
    n_jobs : int, opcional, padrão=-1
        Número de threads para execução paralela. Use -1 para utilizar todos os processadores disponíveis.
    verbose : int, opcional, padrão=1
        Nível de detalhamento do log durante a execução da busca.

    Retorna:
    --------
    list
        Uma lista contendo:
        - run_id: str
          O ID da execução registrada no MLflow.
        - best_model: object
          O melhor estimador obtido pela busca aleatória.
    """

    with mlflow.start_run(run_name=f'Unsupervised_RandomSearchCV_{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        rand_search = RandomizedSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )

        rand_search.fit(X_train)        
        best_model = rand_search.best_estimator_

        mlflow.log_params(rand_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {rand_search.best_params_}')

        return run_id, best_model

def unsupervised_grid_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Realiza uma busca exaustiva de hiperparâmetros em um modelo não supervisionado com registro no MLflow.

    Esta função utiliza `GridSearchCV` para testar todas as combinações possíveis de 
    hiperparâmetros fornecidas no `param_grid`, com base nos dados de treino e validação cruzada. 
    Durante o processo, os parâmetros otimizados são registrados no MLflow para rastreamento de experimentos.

    Parâmetros:
    -----------
    model : object
        O modelo de aprendizado não supervisionado a ser ajustado.
    param_grid : dict
        O espaço de busca contendo todas as combinações possíveis de hiperparâmetros.
    X_train : array-like ou pd.DataFrame
        Dados de treino das variáveis preditoras.
    cv : int, opcional, padrão=5
        Número de divisões para validação cruzada.
    n_jobs : int, opcional, padrão=-1
        Número de threads para execução paralela. Use -1 para utilizar todos os processadores disponíveis.
    verbose : int, opcional, padrão=1
        Nível de detalhamento do log durante a execução da busca.

    Retorna:
    --------
    list
        Uma lista contendo:
        - run_id: str
          O ID da execução registrada no MLflow.
        - best_model: object
          O melhor estimador obtido pela busca exaustiva.
    """

    with mlflow.start_run(run_name=f'Unsupervised_GridSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train)
        
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {grid_search.best_params_}')

        return run_id, best_model

def unsupervised_bayesian_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    """
    Realiza uma busca Bayesiana de hiperparâmetros em um modelo não supervisionado com registro no MLflow.

    Esta função utiliza `BayesSearchCV` para otimizar os hiperparâmetros do modelo fornecido, 
    com base nos dados de treino e validação cruzada. Durante o processo, os parâmetros 
    otimizados são registrados no MLflow para rastreamento de experimentos.

    Parâmetros:
    -----------
    model : object
        O modelo de aprendizado não supervisionado a ser ajustado.
    param_grid : dict
        O espaço de busca para os hiperparâmetros.
    X_train : array-like ou pd.DataFrame
        Dados de treino das variáveis preditoras.
    cv : int, opcional, padrão=5
        Número de divisões para validação cruzada.
    n_jobs : int, opcional, padrão=-1
        Número de threads para execução paralela. Use -1 para utilizar todos os processadores disponíveis.
    verbose : int, opcional, padrão=1
        Nível de detalhamento do log durante a execução da busca.

    Retorna:
    --------
    list
        Uma lista contendo:
        - run_id: str
          O ID da execução registrada no MLflow.
        - best_model: object
          O melhor estimador obtido pela busca Bayesiana.
    """
    
    with mlflow.start_run(run_name=f'Unsupervised_BayesSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id
        
        bayesian_search = BayesSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        bayesian_search.fit(X_train)
        
        best_model = bayesian_search.best_estimator_

        mlflow.log_params(bayesian_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {bayesian_search.best_params_}')

        return run_id, best_model