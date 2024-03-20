# API поиска экспертов

API поиска экспертов - это приложение на базе FastAPI, предназначенное для сопоставления экспертов (пользователей) с проектами на основе их интересов и навыков. Используя комбинацию личных качеств, знаний специализированных программ и описаний проектов, приложение идентифицирует и ранжирует лучшие совпадения, используя векторизацию TF-IDF и косинусное сходство.

## Особенности

- Загрузка и обработка данных пользователей и проектов из файлов Excel.
- Определение наиболее релевантных экспертов для каждого проекта на основе текстового анализа.
- API-эндпоинт для запроса лучших экспертов для конкретного проекта.

## Требования

- Python версии 3.8+
- FastAPI
- Uvicorn (для запуска сервера API)
- Pydantic (для проверки данных)
- Pandas (для манипуляций с данными)
- Scikit-learn (для векторизации TF-IDF и косинусного сходства)

## Установка

1. Убедитесь, что на вашей системе установлен Python версии 3.8 или выше.
2. Склонируйте этот репозиторий или загрузите исходный код.
3. Установите необходимые зависимости, выполнив:

```bash
pip install fastapi uvicorn pydantic pandas scikit-learn

## Использование
1. Подготовьте ваши файлы Excel с данными пользователей и проектов согласно указанному формату.
2. Обновите пути к файлам в скрипте, чтобы они указывали на ваши файлы Excel.
3. Запустите сервер FastAPI, выполнив файл main.py

