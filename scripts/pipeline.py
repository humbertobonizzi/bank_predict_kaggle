import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def retorno_pipeline(x, y, col_num, col_cat_num, col_cat, tipo):

    #Transformador de colunas. Salvar como template. Criar texto com explicação.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #Pesquisar qual o melhor Standard Scaler
        ('scaler', StandardScaler())
    ])

    categorical_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessador = ColumnTransformer(transformers=[
        ('num', numeric_transformer, col_num),
        ('cat_numero', categorical_numerical_transformer, col_cat_num),
        ('categorica', categorical_transformer, col_cat)
    ])

    if tipo == 'regressor':
        treino, teste, resposta_treino, resposta_teste = train_test_split(x, y, random_state = 42, test_size=0.2)
    else:
        treino, teste, resposta_treino, resposta_teste = train_test_split(x, y, random_state = 42, stratify=y)

    #método para treinar com todas as categorias, pra evitar a questão do risco de amostragem
    train_objs_num = len(treino)
    dataset = pd.concat(objs=[treino,teste], axis=0)
    dataset_tratado = preprocessador.fit_transform(dataset)
    treino_tratado = dataset_tratado[:train_objs_num]
    teste_tratado = dataset_tratado[train_objs_num:]

    return treino_tratado, teste_tratado, resposta_treino, resposta_teste

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('STD: ', scores.std())