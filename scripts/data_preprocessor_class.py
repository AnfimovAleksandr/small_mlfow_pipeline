import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import country_converter as coco

class KickstarterPreprocessor:
    def __init__(self):
        self.scalers = {}

    def preprocess_raw_before_scaling(self, data):
        """Предобработка данных + feature engineering"""

        data.drop(['keywords', 'project_id'], axis=1, inplace=True)
        data.dropna(inplace=True)

        #------------------------------------- ДАТЫ -------------------------------------------
        # Даты
        data["launched_at"] = pd.to_datetime(data["launched_at"], unit="s")
        data["created_at"] = pd.to_datetime(data["created_at"], unit="s")
        data["deadline"] = pd.to_datetime(data["deadline"], unit="s")
        data["state_changed_at"] = pd.to_datetime(data["state_changed_at"], unit="s")

        # Год
        data["year_trend"] = data["launched_at"].dt.year - data["launched_at"].dt.year.min()

        # Месяц
        data['month_launched'] = data["launched_at"].dt.month.astype("category")

        # Делаем OHE месяца
        data = pd.get_dummies(
            data,
            columns=["month_launched"],
            prefix="month_launched",        
            prefix_sep="_",        
            drop_first=True        
        )

        # Преобразование дня
        data["day_sin"] = np.sin(2 * np.pi * data["launched_at"].dt.day / 31)
        data["day_cos"] = np.cos(2 * np.pi * data["launched_at"].dt.day / 31)

        # Флаг на выходные
        data["is_weekend"] = data["launched_at"].dt.dayofweek.isin([5, 6]).astype(int)

        # Разница в днях для остальных дат
        data['created_dif'] = (data['launched_at'] - data['created_at']).dt.days
        data['campaign_duration'] = (data['deadline'] - data['launched_at']).dt.days
        data.drop(['state_changed_at', 'launched_at', 'created_at', 'deadline'], axis = 1, inplace=True)

        #------------------------------------- КАТЕГОРИАЛЬНЫЕ -------------------------------------------

        # disable_communication
        data['disable_communication'] = data['disable_communication'].astype(int)

        # Конвертируем коды стран в регионы
        cc = coco.CountryConverter()
        data["region"] = data["country"].apply(
            lambda x: cc.convert(names=x, to="continent", not_found="Other")
        )
        # OHE
        data = pd.get_dummies(
            data,
            columns=["region"],
            prefix="region",        
            prefix_sep="_",        
            drop_first=True        
        )
        data.drop(['country'], axis=1, inplace=True)

        # currency
        main_currencies = set(['USD', 'GBP', 'EUR', 'CAD'])
        data['currency'] = data['currency'].apply(lambda x: x if x in main_currencies else 'Other')
        data = pd.get_dummies(
            data,
            columns=["currency"],
            prefix="currency",        
            prefix_sep="_",        
            drop_first=True        
        )

        #------------------------------------- ЧИСЛОВЫЕ ------------------------------------------- 

        data["goal"] = np.log10(data["goal"] + 1)
        data["backers_count"] = np.log10(data["backers_count"] + 1)
        data["created_dif"] = np.log10(data["created_dif"] + 1)

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

        data["duration_category"] = data["campaign_duration"].apply(categorize_duration)

        data = pd.get_dummies(
            data,
            columns=["duration_category"],
            prefix="campaign_duration",        
            prefix_sep="_"       
        )
        data.drop(['campaign_duration_Normal', 'campaign_duration'], axis=1, inplace=True)

        #------------------------------------- ПРОСТЫЕ ТЕКСТОВЫЕ ------------------------------------------- 

        data['words_name'] = data['name'].apply(lambda x: len(x.split()))
        data['words_desc'] = data['desc'].apply(lambda x: len(x.split()))
        data['uppercase_ratio'] = data['desc'].apply(lambda text: sum(1 for char in text if char.isupper()) / max(len(text), 1))
        data['exlam_and_quest'] = data['desc'].apply(lambda text: sum(1 for char in text if char == '!' or char == '?'))

        data['uppercase_ratio'] = np.log(data[['uppercase_ratio']] + 0.01)

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

        data["exlam_and_quest"] = data["exlam_and_quest"].apply(exlam_and_quest_classifier)

        # В данном минипроекте используем только модели без текста
        data.drop(['name', 'desc'], axis=1, inplace=True)

        return data

    def fit(self, X_train):
        """Обучает преобразователи на предобработанных обучающих данных"""

        # Объявляем scalers для всех численных признаков
        self.scalers['goal'] = RobustScaler()
        self.scalers['backers_count'] = RobustScaler()
        self.scalers['created_dif'] = RobustScaler()
        self.scalers['words_name'] = StandardScaler()
        self.scalers['words_desc'] = StandardScaler()
        self.scalers['uppercase_ratio'] = StandardScaler()

        # Обучаем
        self.scalers['goal'].fit(X_train[["goal"]])
        self.scalers['backers_count'].fit(X_train[["backers_count"]])
        self.scalers['created_dif'].fit(X_train[["created_dif"]])
        self.scalers['words_name'].fit(X_train[["words_name"]])
        self.scalers['words_desc'].fit(X_train[["words_desc"]])
        self.scalers['uppercase_ratio'].fit(X_train[["uppercase_ratio"]])

        return self
    
    def transform(self, X):
        """Применяет преобразования к данным"""

        X['goal'] =  self.scalers['goal'].transform(X[["goal"]])
        X['backers_count'] =  self.scalers['backers_count'].transform(X[["backers_count"]])
        X['created_dif'] =  self.scalers['created_dif'].transform(X[["created_dif"]])
        X['words_name'] =  self.scalers['words_name'].transform(X[["words_name"]])
        X['words_desc'] =  self.scalers['words_desc'].transform(X[["words_desc"]])
        X['uppercase_ratio'] =  self.scalers['uppercase_ratio'].transform(X[["uppercase_ratio"]])

        return X

    def save(self, path):
        """Сохраняет объект предобработки"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """Загружает объект предобработки"""
        return joblib.load(path)