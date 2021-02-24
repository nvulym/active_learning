import pandas as pd
import numpy as np
import random as rd
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools, Descriptors
from sklearn.metrics import balanced_accuracy_score


def get_ba(df, df_train):
    smiles = [Chem.MolFromSmiles(i) for i in df['smiles']]
    Y_test = df['pKi'] >= 7
    X_test = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles])
    smiles_train = [Chem.MolFromSmiles(i) for i in df_train['smiles']]
    Y_train = df_train['pKi'] >= 7
    X_train = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in smiles_train])
    mod = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=1, n_jobs=-1).fit(X_train, Y_train)
    Y_pred_test = mod.predict(X_test)

    return balanced_accuracy_score(Y_test, Y_pred_test)


def get_ba_for_ideal(df_external_test_set, df_training_set, Y_test_external, X_test_external, random_numbers, maxlen,
                     initial_df_train_length):
    df_train = df_training_set.loc[df_training_set.index.intersection(random_numbers),
               :]  # в обучении будет рандомные 10 строк из обучающей выборки
    df_training_set = df_training_set.drop(
        random_numbers)  # в обучающей убираются 10 строк, которые добавились в обучение
    results = dict()  # словарь, где ключ - это количество объектов, значение - это полученная балансированная точность

    initial_ba = get_ba(df_external_test_set, df_train)  # начальное ba при обучающей выборке из 10 строк
    results[initial_df_train_length] = initial_ba  # первоначальное ba записываем в словарь

    df_temp = df_training_set
    len_df_train = len(df_train)  # длина обучающей выборки
    counter = 0  # счетчик
    while len(df_train) != len_df_train + maxlen:  # условие для завершения цикла.
        # пока длина обучающей выборки не достигла нужного
        ba_max_triple = [0, 0, 0]  # 0 - ind, 1 - row, 0 - ba value
        index = df_training_set.index
        ind_ = rd.choices(index, k=100)
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

            ba = balanced_accuracy_score(Y_test_external, Y_pred)  # находим балансированную точность

            df_temp = df_temp.append(row)  # возвращем обратно строку в тестовую
            df_train = df_train.drop(index=ind)  # удаляем из обучающей
            if ba > ba_max_triple[2]:
                ba_max_triple = [ind, row, ba]  # перезаписываем значение

        counter += 1
        df_training_set = df_training_set.drop(ba_max_triple[0])  # если значение перезаписалось, то выполнилось
        # условие максимума, поэтому из исходной окончательно удаляем эту строку (чтобы не натыкались на нее еще раз)
        df_train = df_train.append(ba_max_triple[1])  # а в обучающую добавляем найденную подходящую строку
        results[initial_df_train_length + counter] = get_ba(df_external_test_set, df_train)  # записываем результат
    return results


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
model = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=1, n_jobs=-1).fit(X_train, Y_train)
Y_pred_test = model.predict(X_test_external)
BA = balanced_accuracy_score(Y_test_external, Y_pred_test)
print(BA)

# идеальный случай отбор данных
n = 10
for step in range(7):
    random_numbers = rd.choices(random_numbers_for_training_set, k=n)
    ba_val = get_ba_for_ideal(df_external_test_set, df_training_set, Y_test_external, X_test_external, random_numbers,
                              maxlen=None, initial_df_train_length=n)
    number = ba_val.keys()
    ba = ba_val.values()
    result = pd.DataFrame([number, ba], index=['Number', 'BA']).T
    result.to_csv('FILES_CHECK/CHEBML244_check_{}.csv'.format(step))
