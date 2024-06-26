# Прогнозирование качества вина

Этот проект предназначен для прогнозирования качества вина с использованием моделей машинного обучения. Проект включает скрипты для обучения, предсказания и веб-интерфейс на основе FastAPI.

## Начало работы

Эти инструкции помогут вам настроить и запустить проект на вашем локальном компьютере для разработки и тестирования.

Требования

Убедитесь, что у вас установлены следующие программы:

- Python 3.10
- Docker
- Docker Compose

## Клонирование репозитория

Склонируйте репозиторий с GitHub:

`git clone https://github.com/AjaxRu/Wine_quality.git`  
`cd Wine_quality`


## Вариант 1: Использование Docker
### Шаг 1: Сборка и запуск контейнеров Docker
Запустите Docker Desktop  
Соберите и запустите контейнеры Docker с помощью Docker Compose:  
`docker-compose up --build`  

Эта команда соберет образ Docker и запустит контейнер для вашего приложения.  

### Шаг 2 (опционально): Обучение модели
В отдельном терминале выполните команду для запуска процесса обучения внутри контейнера Docker:  
`docker-compose run train python Wine_quality_model/train_pipeline.py`  

Эта команда выполнит скрипт обучения Wine_quality_model/train_pipeline.py внутри контейнера.  

### Шаг 3: Доступ к приложению
После запуска контейнеров вы можете получить доступ к приложению в веб-браузере по адресу:  
`http://127.0.0.1:8000/`   
На этой странице вы сможете ввести характеристики вина и получить прогноз качества.  

Если вы хотите проверить активные сеансы Docker:  
`docker ps`  
Остановить контейнер(ы):  
`docker stop <container_id>`  
`docker stop $(docker ps -q)`  

Удалить контейнер(ы):  
`docker rm <container_id>`  
`docker rm $(docker ps -a -q)`  


## Вариант 2: Локальная установка и запуск без Docker
### Шаг 1: Создание и активация виртуального окружения
`python -m venv venv`  
`venv\Scripts\activate`  
`pip install -e .`  
### Шаг 2: Установка зависимостей
`pip install -r requirements/requirements.txt`  
`pip install -r requirements/test_requirements.txt`  

### Шаг 3 (опционально): Обучение модели
Выполните команду для обучения модели:  
`python Wine_quality_model/train_pipeline.py`
![image](https://github.com/AjaxRu/Wine_quality/assets/145920622/1cb0e8e4-d5ea-49bf-9709-cbaee48592e6)![image](https://github.com/AjaxRu/Wine_quality/assets/145920622/fb4fe356-760e-43df-85e4-38bc9c47cbdf)


### Шаг 4: Запуск приложения
Выполните команду для запуска FastAPI приложения:  
`python app.py`  

### Шаг 5: Доступ к приложению
Откройте браузер и перейдите по адресу:  
`http://127.0.0.1:8000/`  
На этой странице вы сможете ввести характеристики вина и получить прогноз качества.  
![image](https://github.com/AjaxRu/Wine_quality/assets/145920622/6c56258f-a274-41fb-b917-4d89c56b99e6)

## Запуск тестов
Для запуска тестов используйте следующую команду:  
`pytest`  
