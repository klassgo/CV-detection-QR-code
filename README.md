# QR Code Detection on Conveyor

Прототип системы компьютерного зрения для обнаружения QR-кодов на товарах, движущихся по конвейеру.

Проект выполнен как MVP: нейросеть локализует QR-коды на изображении, после чего найденные области передаются классическому декодеру для считывания содержимого.

## Архитектура

```text
Изображение
    ↓
YOLO26m
    ↓
Bounding boxes QR-кодов
    ↓
ZXing
    ↓
Декодированные значения
```

YOLO используется только для локализации QR-кодов.
Для декодирования применяется zxing-cpp.

## Dataset
  Для обучения использован открытый датасет Qrcode Finder v1 из Roboflow Universe.
  - 3045 изображений
  - 1 класс — QR code
  - готовая YOLO-разметка
  - train / validation / test split
  - изображения приведены к размеру 640×640
  Dataset:
  https://universe.roboflow.com/ironpark/qrcode-finder/dataset/1

## Model
Использована модель YOLO26m с предварительно обученными весами.
YOLO выбрана благодаря:
- хорошему качеству object detection;
- возможности transfer learning;
- простой интеграции через Ultralytics;
- достаточно быстрому inference;
- возможности обучения на Kaggle GPU.

### Обучение выполнялось на Kaggle с разрешением imgsz = 1024 на GPU T4 x2 (~ 1 час)

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/7d1c8b0b-0007-48d8-824a-12a9e7708945" />

 ---
 
  После обучения использовался checkpoint:
  ```
  best.pt
  ```
## Для демонстрации создано веб-приложение на Streamlit.

Ссылка, если хостится - https://cv-detection-qr-code-zxc.streamlit.app

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/3e33f7b7-083c-4a60-a18d-a40f20f5d195" />


Пользователь загружает изображение, после чего приложение:
  1. запускает YOLO26m;
  2. находит QR-коды;
  3. выделяет найденные области;
  4. декодирует QR через ZXing;
  5. отображает результат.

<img width="1920" height="1920" alt="train_batch5361" src="https://github.com/user-attachments/assets/333fc42f-f70d-49c9-ab1e-8911afba5313" />


## Основные файлы:
- app.py          — Streamlit-интерфейс
- reader.py       — YOLO detection + QR decoding
- read_codes.py   — запуск из терминала
- best.pt         — обученные веса
- requirements.txt
  
## Run locally
```
  pip install -r requirements.txt
  streamlit run app.py
```
