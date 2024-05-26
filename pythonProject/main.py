# # import pandas as pd
# # import os
# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC
# # from sklearn.metrics import classification_report, confusion_matrix
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # Проверка наличия файла
# # file_path = 'declension_hatred_cleaned.csv'  # Убедитесь, что путь правильный
# # if not os.path.exists(file_path):
# #     raise FileNotFoundError(f"The file at path {file_path} does not exist.")
# #
# # # Загрузка данных с указанием разделителя и кодировки
# # df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
# #
# # # Вывод первых строк и информации о наборе данных
# # print(df.head())
# # print(df.info())
# #
# # # Удаление строк с пропущенными значениями и дублирующихся строк
# # df_cleaned = df.dropna().drop_duplicates()
# #
# # # Преобразование категориальной переменной `type` в числовую
# # label_encoder = LabelEncoder()
# # df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])
# #
# # # Разделение данных на обучающую и тестовую выборки
# # X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)
# #
# # # Векторизация текста
# # vectorizer = TfidfVectorizer(max_features=1000)
# # X_train_vect = vectorizer.fit_transform(X_train)
# # X_test_vect = vectorizer.transform(X_test)
# #
# # # Определение базовых моделей
# # estimators = [
# #     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
# #     ('svr', SVC(probability=True))
# # ]
# #
# # # Определение мета-модели
# # stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# # rf = RandomForestClassifier(n_estimators=100, random_state=42)
# # gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# #
# # # Обучение моделей
# # stacking.fit(X_train_vect, y_train)
# # rf.fit(X_train_vect, y_train)
# # gb.fit(X_train_vect, y_train)
# #
# # # Предсказания моделей
# # y_pred_stacking = stacking.predict(X_test_vect)
# # y_pred_rf = rf.predict(X_test_vect)
# # y_pred_gb = gb.predict(X_test_vect)
# #
# # # Отчет о классификации
# # print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
# # print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# # print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))
# #
# # # Функция для визуализации матрицы ошибок
# # def plot_confusion_matrix(cm, title, ax):
# #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
# #     ax.set_xlabel('Predicted')
# #     ax.set_ylabel('True')
# #     ax.set_title(title)
# #
# # # Матрица ошибок и визуализация
# # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# #
# # cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# # cm_rf = confusion_matrix(y_test, y_pred_rf)
# # cm_gb = confusion_matrix(y_test, y_pred_gb)
# #
# # plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
# # plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
# # plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])
# #
# # plt.show()

# Импорт необходимых библиотек
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Загрузка данных с указанием разделителя и кодировки
# file_path = 'declension_hatred_cleaned.csv'
# df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
# # Вывод первых строк и информации о наборе данных
# print(df.head())
# print(df.info())
#
# # Удаление строк с пропущенными значениями
# df_cleaned = df.dropna()
#
# # Удаление дублирующихся строк
# df_cleaned = df_cleaned.drop_duplicates()
#
# # Вывод информации о очищенном наборе данных
# print(df_cleaned.info())
#
# # Преобразование категориальной переменной `type` в числовую
# label_encoder = LabelEncoder()
# df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)
#
# # Векторизация текста
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_vect = vectorizer.fit_transform(X_train)
# X_test_vect = vectorizer.transform(X_test)
#
# # Определение базовых моделей
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('svr', SVC(probability=True))
# ]
#
# # Определение мета-модели
# stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#
# # Обучение моделей
# stacking.fit(X_train_vect, y_train)
# rf.fit(X_train_vect, y_train)
# gb.fit(X_train_vect, y_train)
#
# # Предсказания моделей
# y_pred_stacking = stacking.predict(X_test_vect)
# y_pred_rf = rf.predict(X_test_vect)
# y_pred_gb = gb.predict(X_test_vect)
#
# # Отчет о классификации для всех моделей
# print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
# print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))
#
# # Функция для визуализации матрицы ошибок
# def plot_confusion_matrix(cm, title, ax):
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(title)
#
# # Матрица ошибок и визуализация
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# cm_gb = confusion_matrix(y_test, y_pred_gb)
#
# plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
# plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
# plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])
#
# plt.show()
#
# # Оптимизация моделей
#
# # Grid Search для Random Forest
# param_grid_rf = {
#     'n_estimators': [50, 100, 200],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8],
#     'max_depth': [4, 6, 8, 10],
#     'criterion': ['gini', 'entropy']
# }
#
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
# grid_search_rf.fit(X_train_vect, y_train)
# best_rf = grid_search_rf.best_estimator_
#
# print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
# y_pred_best_rf = best_rf.predict(X_test_vect)
# print("Optimized Random Forest:\n", classification_report(y_test, y_pred_best_rf, zero_division=0))
#
# cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_rf, "Confusion Matrix for Optimized Random Forest", ax)
# plt.show()
#
# # Random Search для Gradient Boosting
# param_dist_gb = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5, 6],
#     'subsample': [0.6, 0.8, 1.0],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8]
# }
#
# random_search_gb = RandomizedSearchCV(estimator=gb, param_distributions=param_dist_gb, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
# random_search_gb.fit(X_train_vect, y_train)
# best_gb = random_search_gb.best_estimator_
#
# print("Best parameters for Gradient Boosting: ", random_search_gb.best_params_)
# y_pred_best_gb = best_gb.predict(X_test_vect)
# print("Optimized Gradient Boosting:\n", classification_report(y_test, y_pred_best_gb, zero_division=0))
#
# cm_best_gb = confusion_matrix(y_test, y_pred_best_gb)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_gb, "Confusion Matrix for Optimized Gradient Boosting", ax)
# plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
#
# # Загрузка данных с указанием разделителя и кодировки
# file_path = 'declension_hatred_cleaned.csv'
# df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
# # Вывод первых строк и информации о наборе данных
# print(df.head())
# print(df.info())
#
# # Удаление строк с пропущенными значениями
# df_cleaned = df.dropna()
#
# # Удаление дублирующихся строк
# df_cleaned = df_cleaned.drop_duplicates()
#
# # Вывод информации о очищенном наборе данных
# print(df_cleaned.info())
#
# # Преобразование категориальной переменной `type` в числовую
# label_encoder = LabelEncoder()
# df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)
#
# # Векторизация текста
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_vect = vectorizer.fit_transform(X_train)
# X_test_vect = vectorizer.transform(X_test)
#
# # Определение базовых моделей
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('svr', SVC(probability=True))
# ]
#
# # Определение мета-модели
# stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#
# # Замер времени обучения моделей
# start_time = time.time()
# stacking.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Stacking Classifier training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# gb.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Gradient Boosting training time: {end_time - start_time:.2f} seconds")
#
# # Предсказания моделей
# y_pred_stacking = stacking.predict(X_test_vect)
# y_pred_rf = rf.predict(X_test_vect)
# y_pred_gb = gb.predict(X_test_vect)
#
# # Отчет о классификации для всех моделей
# print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
# print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))
#
# # Функция для визуализации матрицы ошибок
# def plot_confusion_matrix(cm, title, ax):
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(title)
#
# # Матрица ошибок и визуализация
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# cm_gb = confusion_matrix(y_test, y_pred_gb)
#
# plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
# plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
# plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])
#
# plt.show()
#
# # Оптимизация моделей
#
# # Grid Search для Random Forest
# param_grid_rf = {
#     'n_estimators': [50, 100, 200],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8],
#     'max_depth': [4, 6, 8, 10],
#     'criterion': ['gini', 'entropy']
# }
#
# start_time = time.time()
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
# grid_search_rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest Grid Search training time: {end_time - start_time:.2f} seconds")
#
# best_rf = grid_search_rf.best_estimator_
# print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
# y_pred_best_rf = best_rf.predict(X_test_vect)
# print("Optimized Random Forest:\n", classification_report(y_test, y_pred_best_rf, zero_division=0))
#
# cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_rf, "Confusion Matrix for Optimized Random Forest", ax)
# plt.show()
#
# # Random Search для Gradient Boosting
# param_dist_gb = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5, 6],
#     'subsample': [0.6, 0.8, 1.0],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8]
# }
#
# start_time = time.time()
# random_search_gb = RandomizedSearchCV(estimator=gb, param_distributions=param_dist_gb, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
# random_search_gb.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Gradient Boosting Random Search training time: {end_time - start_time:.2f} seconds")
#
# best_gb = random_search_gb.best_estimator_
# print("Best parameters for Gradient Boosting: ", random_search_gb.best_params_)
# y_pred_best_gb = best_gb.predict(X_test_vect)
# print("Optimized Gradient Boosting:\n", classification_report(y_test, y_pred_best_gb, zero_division=0))
#
# cm_best_gb = confusion_matrix(y_test, y_pred_best_gb)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_gb, "Confusion Matrix for Optimized Gradient Boosting", ax)
# plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
#
# # Загрузка данных с указанием разделителя и кодировки
# file_path = 'declension_hatred_cleaned.csv'
# df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
# # Вывод первых строк и информации о наборе данных
# print(df.head())
# print(df.info())
#
# # Удаление строк с пропущенными значениями
# df_cleaned = df.dropna()
#
# # Удаление дублирующихся строк
# df_cleaned = df_cleaned.drop_duplicates()
#
# # Вывод информации о очищенном наборе данных
# print(df_cleaned.info())
#
# # Преобразование категориальной переменной `type` в числовую
# label_encoder = LabelEncoder()
# df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)
#
# # Векторизация текста
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_vect = vectorizer.fit_transform(X_train)
# X_test_vect = vectorizer.transform(X_test)
#
# # Определение базовых моделей
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('svr', SVC(probability=True))
# ]
#
# # Определение мета-модели
# stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#
# # Замер времени обучения моделей
# start_time = time.time()
# stacking.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Stacking Classifier training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# gb.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Gradient Boosting training time: {end_time - start_time:.2f} seconds")
#
# # Предсказания моделей
# y_pred_stacking = stacking.predict(X_test_vect)
# y_pred_rf = rf.predict(X_test_vect)
# y_pred_gb = gb.predict(X_test_vect)
#
# # Отчет о классификации для всех моделей
# print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
# print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))
#
# # Функция для визуализации матрицы ошибок
# def plot_confusion_matrix(cm, title, ax):
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(title)
#
# # Матрица ошибок и визуализация
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# cm_gb = confusion_matrix(y_test, y_pred_gb)
#
# plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
# plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
# plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])
#
# plt.show()
#
# # Оптимизация моделей
#
# # Grid Search для Random Forest
# param_grid_rf = {
#     'n_estimators': [50, 100, 200],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8],
#     'max_depth': [4, 6, 8, 10],
#     'criterion': ['gini', 'entropy']
# }
#
# start_time = time.time()
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
# grid_search_rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest Grid Search training time: {end_time - start_time:.2f} seconds")
#
# best_rf = grid_search_rf.best_estimator_
# print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
# y_pred_best_rf = best_rf.predict(X_test_vect)
# print("Optimized Random Forest:\n", classification_report(y_test, y_pred_best_rf, zero_division=0))
#
# cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_rf, "Confusion Matrix for Optimized Random Forest", ax)
# plt.show()
#
# # Random Search для Gradient Boosting
# param_dist_gb = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5, 6],
#     'subsample': [0.6, 0.8, 1.0],
#     'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8]
# }
#
# start_time = time.time()
# random_search_gb = RandomizedSearchCV(estimator=gb, param_distributions=param_dist_gb, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
# random_search_gb.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Gradient Boosting Random Search training time: {end_time - start_time:.2f} seconds")
#
# best_gb = random_search_gb.best_estimator_
# print("Best parameters for Gradient Boosting: ", random_search_gb.best_params_)
# y_pred_best_gb = best_gb.predict(X_test_vect)
# print("Optimized Gradient Boosting:\n", classification_report(y_test, y_pred_best_gb, zero_division=0))
#
# cm_best_gb = confusion_matrix(y_test, y_pred_best_gb)
# fig, ax = plt.subplots(figsize=(6, 6))
# plot_confusion_matrix(cm_best_gb, "Confusion Matrix for Optimized Gradient Boosting", ax)
# plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
#
# # Загрузка данных с указанием разделителя и кодировки
# file_path = 'declension_hatred_cleaned.csv'
# df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
# # Удаление строк с пропущенными значениями
# df_cleaned = df.dropna()
#
# # Удаление дублирующихся строк
# df_cleaned = df_cleaned.drop_duplicates()
#
# # Преобразование категориальной переменной `type` в числовую
# label_encoder = LabelEncoder()
# df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)
#
# # Векторизация текста
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_vect = vectorizer.fit_transform(X_train)
# X_test_vect = vectorizer.transform(X_test)
#
# # Определение базовых моделей
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('svr', SVC(probability=True))
# ]
#
# # Определение мета-модели
# stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#
# # Замер времени обучения моделей
# start_time = time.time()
# stacking.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Stacking Classifier training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest training time: {end_time - start_time:.2f} seconds")
#
# start_time = time.time()
# gb.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Gradient Boosting training time: {end_time - start_time:.2f} seconds")
#
# # Предсказания моделей
# y_pred_stacking = stacking.predict(X_test_vect)
# y_pred_rf = rf.predict(X_test_vect)
# y_pred_gb = gb.predict(X_test_vect)
#
# # Отчет о классификации для всех моделей
# print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
# print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
# print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))
#
# # Функция для визуализации матрицы ошибок
# def plot_confusion_matrix(cm, title, ax):
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(title)
#
# # Матрица ошибок и визуализация
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# cm_gb = confusion_matrix(y_test, y_pred_gb)
#
# plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
# plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
# plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])
#
# plt.show()
#
# # Оптимизация моделей
#
# # Grid Search для Random Forest
# param_grid_rf = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_features': ['sqrt', 'log2'],
#     'max_depth': [4, 6, 8, 10, 20],
#     'criterion': ['gini', 'entropy']
# }
#
# start_time = time.time()
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
# grid_search_rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest Grid Search training time: {end_time - start_time:.2f} seconds")
#
# best_rf = grid_search_rf.best_estimator_
# print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
#
# # Предсказания и отчет для оптимизированной модели
# y_pred_best_rf = best_rf.predict(X_test_vect)
# print("Optimized Random Forest:\n", classification_report(y_test, y_pred_best_rf, zero_division=0))
#
# # Матрица ошибок и визуализация для оптимизированной модели
# cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# ax.set_title('Confusion Matrix for Optimized Random Forest')
# plt.show()
#
# # Random Search для Random Forest
# param_dist_rf = {
#     'n_estimators': [50, 100, 200, 300, 400, 500],
#     'max_features': ['sqrt', 'log2', None],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'criterion': ['gini', 'entropy'],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# start_time = time.time()
# random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
# random_search_rf.fit(X_train_vect, y_train)
# end_time = time.time()
# print(f"Random Forest Random Search training time: {end_time - start_time:.2f} seconds")
#
# best_rf_random = random_search_rf.best_estimator_
# print("Best parameters for Random Forest (Random Search): ", random_search_rf.best_params_)
#
# # Предсказания и отчет для оптимизированной модели
# y_pred_best_rf_random = best_rf_random.predict(X_test_vect)
# print("Optimized Random Forest (Random Search):\n", classification_report(y_test, y_pred_best_rf_random, zero_division=0))
#
# # Матрица ошибок и визуализация для оптимизированной модели
# cm_best_rf_random = confusion_matrix(y_test, y_pred_best_rf_random)
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.heatmap(cm_best_rf_random, annot=True, fmt='d', cmap='Blues', ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# ax.set_title('Confusion Matrix for Optimized Random Forest (Random Search)')
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Загрузка данных с указанием разделителя и кодировки
file_path = 'declension_hatred_cleaned.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')

# Удаление строк с пропущенными значениями
df_cleaned = df.dropna()

# Удаление дублирующихся строк
df_cleaned = df_cleaned.drop_duplicates()

# Преобразование категориальной переменной `type` в числовую
label_encoder = LabelEncoder()
df_cleaned['type_encoded'] = label_encoder.fit_transform(df_cleaned['type'])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df_cleaned['word'], df_cleaned['type_encoded'], test_size=0.2, random_state=42)

# Векторизация текста с использованием биграмм и триграмм
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Балансировка классов вручную с помощью весов классов
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Определение моделей
rf = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)
gb = GradientBoostingClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# Определение базовых моделей для Stacking Classifier
stacking_estimators = [
    ('rf', rf),
    ('svc', svc)
]

# Определение мета-модели для Stacking Classifier
stacking = StackingClassifier(estimators=stacking_estimators, final_estimator=LogisticRegression())

# Замер времени обучения моделей
start_time = time.time()
stacking.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Stacking Classifier training time: {end_time - start_time:.2f} seconds")

start_time = time.time()
rf.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")

start_time = time.time()
gb.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Gradient Boosting training time: {end_time - start_time:.2f} seconds")

# Предсказания моделей
y_pred_stacking = stacking.predict(X_test_vect)
y_pred_rf = rf.predict(X_test_vect)
y_pred_gb = gb.predict(X_test_vect)

# Отчет о классификации для всех моделей
print("Stacking Classifier:\n", classification_report(y_test, y_pred_stacking, zero_division=0))
print("Random Forest:\n", classification_report(y_test, y_pred_rf, zero_division=0))
print("Gradient Boosting:\n", classification_report(y_test, y_pred_gb, zero_division=0))

# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

# Матрица ошибок и визуализация
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cm_stacking = confusion_matrix(y_test, y_pred_stacking)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_gb = confusion_matrix(y_test, y_pred_gb)

plot_confusion_matrix(cm_stacking, "Confusion Matrix for Stacking Classifier", axes[0])
plot_confusion_matrix(cm_rf, "Confusion Matrix for Random Forest", axes[1])
plot_confusion_matrix(cm_gb, "Confusion Matrix for Gradient Boosting", axes[2])

plt.show()

# Оптимизация моделей

# Grid Search для Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10, 20],
    'criterion': ['gini', 'entropy']
}

start_time = time.time()
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Random Forest Grid Search training time: {end_time - start_time:.2f} seconds")

best_rf = grid_search_rf.best_estimator_
print("Best parameters for Random Forest: ", grid_search_rf.best_params_)

# Предсказания и отчет для оптимизированной модели
y_pred_best_rf = best_rf.predict(X_test_vect)
print("Optimized Random Forest:\n", classification_report(y_test, y_pred_best_rf, zero_division=0))

# Матрица ошибок и визуализация для оптимизированной модели
cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix for Optimized Random Forest')
plt.show()

# Random Search для Random Forest
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

start_time = time.time()
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search_rf.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Random Forest Random Search training time: {end_time - start_time:.2f} seconds")

best_rf_random = random_search_rf.best_estimator_
print("Best parameters for Random Forest (Random Search): ", random_search_rf.best_params_)

# Предсказания и отчет для оптимизированной модели
y_pred_best_rf_random = best_rf_random.predict(X_test_vect)
print("Optimized Random Forest (Random Search):\n", classification_report(y_test, y_pred_best_rf_random, zero_division=0))

# Матрица ошибок и визуализация для оптимизированной модели
cm_best_rf_random = confusion_matrix(y_test, y_pred_best_rf_random)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_best_rf_random, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix for Optimized Random Forest (Random Search)')
plt.show()

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf_random),
    ('gb', gb),
    ('stacking', stacking)
], voting='soft')

start_time = time.time()
voting_clf.fit(X_train_vect, y_train)
end_time = time.time()
print(f"Voting Classifier training time: {end_time - start_time:.2f} seconds")

# Предсказания и отчет для Voting Classifier
y_pred_voting = voting_clf.predict(X_test_vect)
print("Voting Classifier:\n", classification_report(y_test, y_pred_voting, zero_division=0))

# Матрица ошибок и визуализация для Voting Classifier
cm_voting = confusion_matrix(y_test, y_pred_voting)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix for Voting Classifier')
plt.show()

