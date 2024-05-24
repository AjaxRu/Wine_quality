# Используем официальный образ Python в качестве базового
FROM python:3.10-slim

# Устанавливаем зависимости для системных библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы с зависимостями в контейнер
COPY requirements/requirements.txt requirements.txt
COPY requirements/test_requirements.txt test_requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r test_requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Указываем команду для запуска приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
