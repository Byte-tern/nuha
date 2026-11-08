1. Подготовка - через Терминал в папке проекта

py -3.12 -m venv venv && venv\Scripts\activate          # Linux: `python3.12 -m venv venv && source venv/bin/activate`. При ошибке (блок системой защиты) `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`. Если не принимает `&&` - заменить на `;`
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132    # для NVIDIA GPU, pip install torch torchvision для CPU
pip install -r requirements.txt
Также в папку проекта надо скачать видео

Зависимости:
ultralytics - детектор (YOLO11-pose) и трекер (BoT-SORT), веса подгружаются при первом запуске
torch/torchvision - разметка
opencv-python - чтение видео, гомография, маски, отрисовка
numpy - вычисления
scipy - венгерский алгоритм (linear_sum_assignment)
Pillow - синхронный вывод видео
Стандартные библиотеки: argparse, pickle, json, os, sys, time, queue, threading, inspect, bisect, subprocess, shutil, tempfile, concurrent.futures, tkinter
Модели и веса: yolo11-pose.pt - автоматически подгружается при запуске

2. Калибровка данных

python calibrate.py k --image koef.png                # k1 по прямым линиям; [c] подбор, [s] сохранить, [q]/Esc выйти, рядом окно с работой коэффициента
python calibrate.py walkable --plan officePlan.jpg    # полигон проходимой области (m - режим "препятствие")
python calibrate.py points --camera 1 --root .        # для камер 1..5: от 8 пар "точка пола в кадре -> точка плана"
python calibrate.py mask --camera 1 --root .          # маска на телевизор и зеркала
python calibrate.py zoom --camera 3 --root .          # зона детального поиска далёких людей на камере 3 (проблемы с детекцией без него)
python calibrate.py check --root .                    # проверка полигонов: calib_check_cam*.jpg

3. Детекция + локальный трекинг. Выполняется однократно. Для пробы: --max-frames 300

python stage1_detect_track.py --root . --out local_tracks.pkl 

4. Склейка камер

python fusion.py --inp local_tracks.pkl --out fusion_result.pkl

5. Запуск интерфейса

python app.py --root . --result fusion_result.pkl --plan officePlan.jpg

6. Настройка качества

- Ошибки определения fps: `--fps-override 3:15`. На разметке `--target-fps` указывает желаемое fps каждой камеры, по умолчанию медиана
- Далёкие люди не детектируются: `calibrate.py zoom --imgsz 1600 --conf 0.1`.
- Один человек -> два ID: `fusion.py --k-gate 3.5 --k-merge 3`. Стоит перекалибровать соответствующую камеру, добавив больше точек
- Два разных человека склеились: уменьшить `--k-gate`, `--k-merge`, `--r0`.
- Остальные параметры - редактировать класс `P` в `fusion.py`.

7. GitHub

User device:

git clone https://github.com/Byte-tern/nuha.git

8. Описание решения

См. презентацию (папка presentation)