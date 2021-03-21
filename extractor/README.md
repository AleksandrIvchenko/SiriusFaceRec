# How to train

- Install module: `./scripts/install_module.sh`
- Download data:
    https://drive.google.com/file/d/1F96x4LDbsTZGMMq81fZr7aduJCe8N95O/view?usp=sharing
    Скачать отсюда, положить в папку `./data`, там сделать `unzip`. Надо проверить, что всё конструктивное (папка с картинками, файлы меток) лежат в `./data/celeba/celeba/`.
- Put Neptune API token in `neptune_api_token.txt`
- Run training process: `python scripts/fit_arcface_model.py`

