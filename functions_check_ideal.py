import pandas as pd
import numpy as np
import random as rd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, accuracy_score, f1_score, \
    fbeta_score, precision_recall_fscore_support, recall_score, auc, average_precision_score
from rdkit import Chem
from rdkit.Chem import AllChem
from enum import Enum


def clusters(X_train, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster).fit(X_train)
    list_ = [[] for i in range(n_cluster)]  # создается список пустых списков, количество списоков = n_cluster
    for j in range(n_cluster):
        for n, el in enumerate(kmeans.labels_):
            if el == j:
                list_[j].append(n)
    numbers = [rd.choice(el) for el in list_]  # радомно выбираются номера из каждого списка-кластера
    # cluster_centers = kmeans.cluster_centers_
    return numbers

# если код по отбору данных через кластеризацию верен, то отбираются разные номера реакций

def prepare_data_for_metric(df, df_train):
    smiles = [Chem.MolFromSmiles(i) for i in df['smiles']]
    Y_test = df['pKi'] >= 7
    X_test = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles])
    smiles_train = [Chem.MolFromSmiles(i) for i in df_train['smiles']]
    Y_train = df_train['pKi'] >= 7
    X_train = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles_train])
    mod = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=1, n_jobs=-1).fit(X_train, Y_train)
    Y_pred_test = mod.predict(X_test)
    return Y_test, Y_pred_test

# метрики классификационной модели
class Metric(Enum):
    CM = 1
    PPV = 2
    BA = 3
    PNV = 4
    ACC = 5
    F_1 = 6
    F_B = 7
    REC_FS = 8
    REC_SC = 9
    ALL = 10
    ROC = 11
    AUC = 12
    AVERAGE_PS = 13


def get_metrics_of_classification_model(Y_test, Y_pred_test, metric, Y_score=None):
    """ Функция, которая считает метрики для классификационной модели """
    if metric == Metric.CM:
        return confusion_matrix(Y_test, Y_pred_test).ravel()  # tn, fp, fn, tp
    elif metric == Metric.PPV:
        return precision_score(Y_test, Y_pred_test)  # та же формула для нахождения: (tp)/(tp + fp)
    elif metric == Metric.BA:
        return balanced_accuracy_score(Y_test, Y_pred_test)
    elif metric == Metric.PNV:
        tn, _, fn, _ = confusion_matrix(Y_test, Y_pred_test).ravel()
        return float(tn) / (tn + fn)
    elif metric == Metric.ACC:
        return accuracy_score(Y_test, Y_pred_test)
    elif metric == Metric.F_1:
        return f1_score(Y_test, Y_pred_test)
    elif metric == Metric.F_B:
        return fbeta_score(Y_test, Y_pred_test, beta=0.5)
    elif metric == Metric.REC_FS:
        return precision_recall_fscore_support(Y_test, Y_pred_test)
    elif metric == Metric.REC_SC:
        return recall_score(Y_test, Y_pred_test)

    all_metrics = []

    if Y_score is not None:
        if metric == Metric.ROC:
            return roc_curve(Y_test, Y_score)  # fpr, tpr, thresholds
        elif metric == Metric.AUC:
            fpr, tpr, _ = roc_curve(Y_test, Y_score)
            return auc(fpr, tpr)
        elif metric == Metric.AVERAGE_PS:
            return average_precision_score(Y_test, Y_score)
        elif metric == Metric.ALL:
            all_metrics.append(get_metrics_of_classification_model(Y_test, Y_pred_test, Metric.ROC, Y_score))
            all_metrics.append(get_metrics_of_classification_model(Y_test, Y_pred_test, Metric.AUC, Y_score))
            all_metrics.append(get_metrics_of_classification_model(Y_test, Y_pred_test, Metric.AVERAGE_PS, Y_score))
        else:
            raise ValueError('Unknown metric')

    if metric == Metric.ALL:
        for m in list(Metric):
            if m != Metric.ROC and m != Metric.AUC and m != Metric.AVERAGE_PS and m != Metric.ALL:
                all_metrics.append(get_metrics_of_classification_model(Y_test, Y_pred_test, m, Y_score))
    return all_metrics


def if_non_valid_raise(obj):
    """ Функция, которая ломает все, если объект является кортежем """ # CONFUSION_MATRIX, ROC_CURVE
    if isinstance(obj, tuple):
        raise TypeError(f'{obj} is tuple')
    if isinstance(obj, list):
        raise TypeError(f'{obj} is list')

def get_metric_for_ideal(name, df_external_test_set, df_training_set, Y_test_external, X_test_external, random_numbers,
                         maxlen, initial_df_train_length, metric):
    df_train = df_training_set.loc[df_training_set.index.intersection(random_numbers),
               :]  # в обучении будет рандомные 10 строк из обучающей выборки
    df_training_set = df_training_set.drop(
        random_numbers)  # в обучающей убираются 10 строк, которые добавились в обучение
    results = [[], []]  # 0 - number, 1 - res_metric

    initial_val = get_metrics_of_classification_model(*(prepare_data_for_metric(df_external_test_set, df_training_set)),
                                                      metric)
    if_non_valid_raise(initial_val)

    results[0].append(initial_df_train_length)
    results[1].append(initial_val)

    df_temp = df_training_set
    len_df_train = len(df_train)  # длина обучающей выборки
    counter = 0  # счетчик
    while len(df_train) != len_df_train + maxlen:  # условие для завершения цикла.
        # пока длина обучающей выборки не достигла нужного
        val_max_triple = [0, 0, 0]  # 0 - ind, 1 - row, 2 - value
        index = df_training_set.index
        ind_ = rd.choices(index, k=1500)  # k - указываем сколько строчек отбираем случайным образом (половина от выборки)
        df_temp = df_training_set.loc[ind_, :]
        for ind, row in df_temp.iterrows():
            df_temp = df_temp.drop(index=ind)  # удаляем строку по индексу из исходной обучающей выборки
            df_train = df_train.append(row)  # добавляем строку
            smiles_train = [Chem.MolFromSmiles(i) for i in df_train['smiles']]
            Y_train_new = df_train['pKi'] >= 7
            X_train_new = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles_train])
            mod = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=1, n_jobs=-1).fit(
                X_train_new, Y_train_new)

            Y_pred = mod.predict(X_test_external)

            val = get_metrics_of_classification_model(Y_test_external, Y_pred, metric)
            if_non_valid_raise(val)
            df_temp = df_temp.append(row)  # возвращем обратно строку в тестовую
            df_train = df_train.drop(index=ind)  # удаляем из обучающей
            if val > val_max_triple[2]:
                val_max_triple = [ind, row, val]  # перезаписываем значение

        counter += 1
        df_training_set = df_training_set.drop(val_max_triple[0])  # если значение перезаписалось, то выполнилось
        # условие максимума, поэтому из исходной окончательно удаляем эту строку (чтобы не натыкались на нее еще раз)
        df_train = df_train.append(val_max_triple[1])  # а в обучающую добавляем найденную подходящую строку
        val_res = get_metrics_of_classification_model(*(prepare_data_for_metric(df_external_test_set, df_train)),
                                                      metric)
        if_non_valid_raise(val_res)
        results[1].append(val_res)  # записываем результат
        results[0].append(initial_df_train_length + counter)
        # НЕ ПРИДУМАЛА, КАК ПРИКРУТИТЬ СЮДА СМАЙЛСЫ
        # 160-161 строчки позволяют после каждой итерации в цикле записывать полученные результаты в csv
        # res = pd.DataFrame([results[0], results[1]], columns=['number', str(metric)]).T
        # res.to_csv(f'FILES_CHECK/{name}_ideal.csv', mode='a')
    return results

def merge_metrics(name, metrics, df_external_test_set, df_training_set, Y_test_external, X_test_external,
                  random_numbers, maxlen, initial_df_train_length):
    """ Функция, которая позволяет объединить окончательные результаты расчетов в одну таблицу  """
    # работает при условии, что в прошлой функции нет строчек 160-161 !!!
    df_body = []
    df_columns = []
    for i in range(len(metrics)):
        df_columns.append(str(metrics[i]))
        res_metric = get_metric_for_ideal(df_external_test_set, df_training_set, Y_test_external, X_test_external,
                                          random_numbers, maxlen, initial_df_train_length, metrics[i])
        if i == 0:
            df_body.append(res_metric[0], res_metric[1])
        else:
            df_body.append(res_metric[1])
    res = pd.DataFrame(df_body, columns=['number', *df_columns]).T
    res.to_csv(f'FILES_CHECK/{name}_ideal.csv', mode='a')
    # return res

# внешняя тестовая выборка
df_external_test_set = pd.read_csv('CHEMBL244_external_test_set.csv')
random_numbers_for_external_test_set = df_external_test_set.iloc[:, 0].values
df_external_test_set.index = random_numbers_for_external_test_set
smiles_external = [Chem.MolFromSmiles(i) for i in df_external_test_set['smiles']]
Y_test_external = df_external_test_set['pKi'] >= 7
X_test_external = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles_external])
# обучающая выборка
df_training_set = pd.read_csv('CHEMBL244_training_set.csv')
random_numbers_for_training_set = list(df_training_set.iloc[:, 0].values)
df_training_set.index = random_numbers_for_training_set
smiles_train = [Chem.MolFromSmiles(i) for i in df_training_set['smiles']]
Y_train = df_training_set['pKi'] >= 7
X_train = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles_train])

# RFCLASSIFIER
del smiles_external, random_numbers_for_external_test_set, smiles_train, Y_train

# все метрики классификационной модели. по данной функции можно вытащить любую метрику, которая нам нужна
all_metric = get_metrics_of_classification_model(*(prepare_data_for_metric(df_external_test_set, df_training_set)),
                                                 Metric.ALL)
# идеальный случай отбор данных
name = 'CHEMBL244'
for step in range(7):
    random_numbers = rd.choices(random_numbers_for_training_set, k=10)  # рандомно отбираются 10 строчек
    ba_val = (df_external_test_set, df_training_set, Y_test_external, X_test_external, random_numbers,
              100, 10, Metric.BA)
    res = pd.DataFrame([ba_val[0], ba_val[1]], columns=['number', str(Metric.BA)]).T
    res.to_csv(f'FILES_CHECK/{name}_ideal_{Metric.BA}_{step}.csv')

# отбор данных через кластеризацию
d = clusters(X_train, n_cluster=5)
random_nums = [random.choice(x) for x in d]
result = merge_metrics(name, [Metric.BA, Metric.PPV], df_external_test_set, df_training_set,
                       Y_test_external, X_test_external, random_nums, 1500, 5)