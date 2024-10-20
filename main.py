import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Чтение данных из файла
data = pd.read_csv('iris.txt', delimiter='\t', header=0)

# Разделение на признаки и метки классов
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Кодирование меток классов
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Замена запятых на точки в значениях признаков
X_train = X_train.replace(',', '.', regex=True)
X_test = X_test.replace(',', '.', regex=True)

# Преобразование значений признаков в числовой формат
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание SVM-классификатора
svm_classifier = SVC()

# Определение параметров для Grid Search
parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'C': [0.1, 1, 10, 100]}

# Подбор параметров с помощью Grid Search
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=parameters, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Наилучшие параметры
best_params = grid_search.best_params_
print("Наилучшие параметры:", best_params)

# Обучение SVM-классификатора с наилучшими параметрами
best_svm_classifier = SVC(**best_params)
best_svm_classifier.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = best_svm_classifier.predict(X_test)

# Оценка производительности
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Точность:", accuracy)
print("Точность (Precision):", precision)
print("Полнота (Recall):", recall)
print("F1-мера (F1 Score):", f1)

# Оценка числа опорных векторов
num_support_vectors = best_svm_classifier.n_support_
print("Количество опорных векторов:", num_support_vectors)
