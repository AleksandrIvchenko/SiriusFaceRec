# Face Recognition. Сервис для распознавания лиц
Сириус, 2021

### Архитектура
- Модель 1: детектор - RetinaFace
- Модель 2: экстрактор - ResNet50 + Arcface + CrossEntropyLoss

### Контейнеры
1) triton: NVIDIA Triton Inference Server https://developer.nvidia.com/nvidia-triton-inference-server
2) web: Fast-API с SQLite https://fastapi.tiangolo.com/
3) telegram: телеграм бот https://python-telegram-bot.org/

### Установка
1) Docker https://docs.docker.com/engine/install/
2) Docker nvidia runtime https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
3) Docker compose https://docs.docker.com/compose/install/

### Запуск проекта:
`sudo docker-compose up --build`

##### База данных лежит в `data\sqlite.db`. Обнуление базы данных:
`sqlite3 sqlite.db < schema.sql`
