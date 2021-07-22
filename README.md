# monsters-corporation

## О приложении

Данный репозиторий представляет собой проектную работу для школы [Intel CV Summer Camp](https://github.com/itlab-vision/CV-SUMMER-CAMP-2021). Тема проекта - создание игры по мотивам мультфильма "Корпорация монстров", включающей в себя:
- Использование детектора эмоций по лицу человека для распозвавания смеха
- Накопление "энергии" от смеха
- Распознавание смеха по аудио

Основываясь на этих идеях, было создано приложение MemCheck, позволяющее проверить качество мемов путем детектирования реакции человека на них.

## Требования для запуска

Требования приведены в соответствии с конфигурацией, на которой проводился запуск:
- OpenVINO 2021.4
- Python 3.8
- numpy 1.21
- python-sounddevice 0.3.15

## Запуск приложения
Для запуска используется командная строка:
```sh
python main.py -i <папка с изображениями для проверки>
```

Пример:
```sh
python main.py -i test-memes
```
Для выхода из приложения нажать **q**. Для переключения картинок нажимать **n**


## Структура репозитория
- main.py - основной файл приложения
- src - содержит в себе исходный код
- models - натренированные модели
- images - изображения для интерфейса приложения
- test-memes - тестовые изображения для проверки работы программы

## Реализация

В приложении использован фреймворк OpenVINO для вывода нейронных сетей. Следующие модели были использованы:
- [aclnet](https://docs.openvinotoolkit.org/2021.1/omz_models_public_aclnet_aclnet.html) - для распознавания смеха по звуку
- [face-detection-adas-0001](https://docs.openvinotoolkit.org/2021.2/omz_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html) - сеть для детектирования лица
- [emotions-recognition-retail-0003](https://docs.openvinotoolkit.org/latest/omz_models_model_emotions_recognition_retail_0003.html) - классификационная модель для распознания улыбки на лице

## Демо
