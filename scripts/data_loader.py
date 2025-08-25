import pandas as pd
import numpy as np
import re
import string

import country_converter as coco

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# Здесь будет ссылка на оригинальный гитхаб с полным проектом

def main():
    data_train = pd.read_csv('data/raw_train.csv')

    # Исправление названий проектов
    data_train['name'] = data_train['keywords'].str.split('-').str.join(' ')
    data_train.drop(['keywords', 'project_id'], axis=1, inplace=True)

    data_train.dropna(inplace=True)
    #------------------------------------- ДАТЫ -------------------------------------------
    # Даты
    data_train["launched_at"] = pd.to_datetime(data_train["launched_at"], unit="s")
    data_train["created_at"] = pd.to_datetime(data_train["created_at"], unit="s")
    data_train["deadline"] = pd.to_datetime(data_train["deadline"], unit="s")
    data_train["state_changed_at"] = pd.to_datetime(data_train["state_changed_at"], unit="s")

    # Год
    data_train["year_trend"] = data_train["launched_at"].dt.year - data_train["launched_at"].dt.year.min()

    # Месяц
    data_train['month_launched'] = data_train["launched_at"].dt.month.astype("category")

    # Делаем OHE месяца
    data_train = pd.get_dummies(
        data_train,
        columns=["month_launched"],
        prefix="month_launched",        
        prefix_sep="_",        
        drop_first=True        
    )

    # Преобразование дня
    data_train["day_sin"] = np.sin(2 * np.pi * data_train["launched_at"].dt.day / 31)
    data_train["day_cos"] = np.cos(2 * np.pi * data_train["launched_at"].dt.day / 31)

    # Флаг на выходные
    data_train["is_weekend"] = data_train["launched_at"].dt.dayofweek.isin([5, 6]).astype(int)

    # Разница в днях для остальных дат
    data_train['created_dif'] = (data_train['launched_at'] - data_train['created_at']).dt.days
    data_train['campaign_duration'] = (data_train['deadline'] - data_train['launched_at']).dt.days
    data_train.drop(['state_changed_at', 'launched_at', 'created_at', 'deadline'], axis = 1, inplace=True)

    #------------------------------------- КАТЕГОРИАЛЬНЫЕ -------------------------------------------

    # disable_communication
    data_train['disable_communication'] = data_train['disable_communication'].astype(int)

    # Конвертируем коды стран в регионы
    cc = coco.CountryConverter()
    data_train["region"] = data_train["country"].apply(
        lambda x: cc.convert(names=x, to="continent", not_found="Other")
    )
    # OHE
    data_train = pd.get_dummies(
        data_train,
        columns=["region"],
        prefix="region",        
        prefix_sep="_",        
        drop_first=True        
    )
    data_train.drop(['country'], axis=1, inplace=True)

    # currency
    main_currencies = set(['USD', 'GBP', 'EUR', 'CAD'])
    data_train['currency'] = data_train['currency'].apply(lambda x: x if x in main_currencies else 'Other')
    data_train = pd.get_dummies(
        data_train,
        columns=["currency"],
        prefix="currency",        
        prefix_sep="_",        
        drop_first=True        
    )

    #------------------------------------- РАЗБИЕНИЕ TRAIN/TEST -------------------------------------------
    y = data_train['final_status'].copy()
    X = data_train.drop(['final_status'], axis= 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #------------------------------------- ЧИСЛОВЫЕ ------------------------------------------- 

    # goal
    # Логарифимруем
    X_train["goal"] = np.log10(X_train["goal"] + 1)
    X_test["goal"] = np.log10(X_test["goal"] + 1)

    # Нормализуем
    scaler = RobustScaler()
    X_train["goal"] = scaler.fit_transform(X_train[["goal"]])
    X_test["goal"] = scaler.transform(X_test[["goal"]])

    # backers_count
    # Логарифимруем
    X_train["backers_count"] = np.log10(X_train["backers_count"] + 1)
    X_test["backers_count"] = np.log10(X_test["backers_count"] + 1)

    # Нормализуем
    scaler = RobustScaler()
    X_train["backers_count"] = scaler.fit_transform(X_train[["backers_count"]])
    X_test["backers_count"] = scaler.transform(X_test[["backers_count"]])

    # created_dif
    # Логарифимруем
    X_train["created_dif"] = np.log10(X_train["created_dif"] + 1)
    X_test["created_dif"] = np.log10(X_test["created_dif"] + 1)

    # Нормализуем
    scaler = RobustScaler()
    X_train["created_dif"] = scaler.fit_transform(X_train[["created_dif"]])
    X_test["created_dif"] = scaler.transform(X_test[["created_dif"]])

    # campaign_duration
    def categorize_duration(days):
        if days <= 14:
            return "Very_Short"
        elif 15 <= days <= 24:
            return "Short"
        elif 25 <= days <= 35:
            return "Normal"
        elif 36 <= days <= 54:
            return "Long"
        elif 55 <= days <= 65:
            return "Two_month"
        else:
            return "Very_long"

    X_train["duration_category"] = X_train["campaign_duration"].apply(categorize_duration)
    X_test["duration_category"] = X_test["campaign_duration"].apply(categorize_duration)

    X_train = pd.get_dummies(
        X_train,
        columns=["duration_category"],
        prefix="campaign_duration",        
        prefix_sep="_"       
    )
    X_train.drop(['campaign_duration_Normal', 'campaign_duration'], axis=1, inplace=True)

    X_test = pd.get_dummies(
        X_test,
        columns=["duration_category"],
        prefix="campaign_duration",        
        prefix_sep="_"     
    )
    X_test.drop(['campaign_duration_Normal', 'campaign_duration'], axis=1, inplace=True)

    #------------------------------------- ПРОСТЫЕ ТЕКСТОВЫЕ ------------------------------------------- 

    X_train['words_name'] = X_train['name'].apply(lambda x: len(x.split()))
    X_train['words_desc'] = X_train['desc'].apply(lambda x: len(x.split()))
    X_train['uppercase_ratio'] = X_train['desc'].apply(lambda text: sum(1 for char in text if char.isupper()) / max(len(text), 1))
    X_train['exlam_and_quest'] = X_train['desc'].apply(lambda text: sum(1 for char in text if char == '!' or char == '?'))

    X_test['words_name'] = X_test['name'].apply(lambda x: len(x.split()))
    X_test['words_desc'] = X_test['desc'].apply(lambda x: len(x.split()))
    X_test['uppercase_ratio'] = X_test['desc'].apply(lambda text: sum(1 for char in text if char.isupper()) / max(len(text), 1))
    X_test['exlam_and_quest'] = X_test['desc'].apply(lambda text: sum(1 for char in text if char == '!' or char == '?'))

    # Количество слов в названии
    scaler = StandardScaler()
    X_train['words_name'] = scaler.fit_transform(X_train[['words_name']])
    X_test['words_name'] = scaler.transform(X_test[['words_name']])

    # Количество слов в описании
    scaler = StandardScaler()
    X_train['words_desc'] = scaler.fit_transform(X_train[['words_desc']])
    X_test['words_desc'] = scaler.transform(X_test[['words_desc']])

    # Доля заглавных букв в описании
    X_train['uppercase_ratio'] = np.log(X_train[['uppercase_ratio']] + 0.01)
    X_test['uppercase_ratio'] = np.log(X_test[['uppercase_ratio']] + 0.01)

    scaler = StandardScaler()
    X_train['uppercase_ratio'] = scaler.fit_transform(X_train[['uppercase_ratio']])
    X_test['uppercase_ratio'] = scaler.transform(X_test[['uppercase_ratio']])

    # Кол-во восклицательных и вопросительных знаков
    def exlam_and_quest_classifier(counter):
        if counter == 0:
            return 0
        elif counter < 4:
            return 1
        elif counter < 9:
            return 2
        else:
            return 3

    X_train["exlam_and_quest"] = X_train["exlam_and_quest"].apply(exlam_and_quest_classifier)

    # В данном минипроекте используем только модели без текста
    X_train_no_text = X_train.drop(['name', 'desc'], axis=1)
    X_test_no_text = X_test.drop(['name', 'desc'], axis=1)

    pd.concat([X_train_no_text, pd.DataFrame(y_train)], axis=1).to_csv('data/train.csv', index=False)
    pd.concat([X_test_no_text, pd.DataFrame(y_test)], axis=1).to_csv('data/test.csv', index=False)

main()