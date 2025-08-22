# Heart Risk FastAPI Service

### Как это работает

Проект состоит из двух основных частей:
1.  **Jupyter Notebook (`notebooks/comprehensive_analysis.ipynb`)**: Это "исследовательская лаборатория". Здесь происходит весь анализ данных, построение пайплайна предобработки, сравнение моделей и обучение лучшей из них. Результатом работы ноутбука является один файл-артефакт: `artifacts/best_pipeline.pkl`.
2.  **FastAPI приложение (`app/main.py`)**: Это сервис, который использует готовый артефакт для получения предсказаний. Он ничего не обучает, а только загружает `best_pipeline.pkl` и применяет его к новым данным.

### Установка

1.  Клонируйте репозиторий:
    ```bash
    git clone <your-repo-url>
    cd fast_api_ai
    ```

2.  Создайте и активируйте виртуальное окружение:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

### Шаг 1: Обучение модели

Весь процесс обучения и анализа находится в ноутбуке.
- Откройте и выполните все ячейки в `notebooks/comprehensive_analysis.ipynb`.

После успешного выполнения в директории `artifacts/` будет создан файл `best_pipeline.pkl`, а в корне проекта - `submission_advanced.csv`.

### Шаг 2: Получение предсказаний

Есть два способа получить предсказания: через CLI-скрипт или через запущенный веб-сервис.

#### Вариант А: Через CLI-скрипт (локально)

Этот скрипт использует тот же `PredictorService`, что и FastAPI, но запускает его локально без поднятия веб-сервера.

```bash
python scripts/predict_file.py \
    --test-file ./heart_test.csv \
    --artifacts-dir ./artifacts \
    --output-file ./submission_cli.csv
```

#### Вариант Б: Через FastAPI сервис

1.  **Запустите приложение:**
    ```bash
    uvicorn app.main:app --reload --port 8010
    ```

2.  **Отправьте запрос** (в другом терминале):
    Скрипт `predict_file.py` можно использовать и для отправки запросов к запущенному сервису.
    ```bash
    python scripts/predict_file.py \
        --test-file ./heart_test.csv \
        --output-file ./submission_api.csv \
        --use-api \
        --api-url http://127.0.0.1:8010
    ```

3.  **Или используйте веб-интерфейс:**
    Откройте в браузере `http://127.0.0.1:8010/` для загрузки файла через форму.

### Docker

Проект также можно запустить в Docker-контейнере.

1.  **Соберите образ:**
    (Убедитесь, что артефакт `artifacts/best_pipeline.pkl` уже сгенерирован)
    ```bash
    make build
    ```

2.  **Запустите контейнер:**
    ```bash
    make run
    ```
    Сервис будет доступен по адресу `http://localhost:8010`.
