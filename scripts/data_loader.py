import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessor_class import KickstarterPreprocessor

# Здесь будет ссылка на оригинальный гитхаб с полным проектом

def main():
    data_train = pd.read_csv('data/raw_train.csv')

    #------------------------------------- Первичная предобработка (без скелеров) -------------------------------------------

    preprocessor = KickstarterPreprocessor()
    data_train = preprocessor.preprocess_raw_before_scaling(data_train)

    #------------------------------------- РАЗБИЕНИЕ TRAIN/TEST -------------------------------------------------------------

    y = data_train['final_status'].copy()
    X = data_train.drop(['final_status'], axis= 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #------------------------------------- Вторичная предобработка (скелеры) ------------------------------------------------

    preprocessor = preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Сохраняем предобработчик
    preprocessor.save("models/preprocessor.pkl")   

    pd.concat([X_train, pd.DataFrame(y_train)], axis=1).to_csv('data/train.csv', index=False)
    pd.concat([X_test, pd.DataFrame(y_test)], axis=1).to_csv('data/test.csv', index=False)

main()