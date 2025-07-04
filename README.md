# Scetch to Renderer

## Подготовка окружения (Windows 11, Python 3.10+, CUDA 12.8)

### Требования
1. CUDA Toolkit 12.8 и драйверы NVIDIA
2. Установленный Python 3.10, 3.11 или 3.12
3. Активное виртуальное окружение и установленный PyTorch с поддержкой CUDA

### Шаги
0. Убедитесь, что CUDA доступна:
   ```cmd
   nvidia-smi
   nvcc --version
   ```
1. Создайте виртуальное окружение:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
2. Обновите `pip`:
   ```cmd
   python -m pip install --upgrade pip
   ```
3. Установите зависимости проекта:
   ```cmd
   pip install -i https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r requirements.txt
   ```
   >Если присутствует файл `requirements.lock`:
   >```cmd
   >pip install -r requirements.lock
   >```
4. Проверки:
   ```cmd
   python -c "import torch; print(torch.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"
   ```
5. Сохраните список установленных пакетов:
   ```cmd
   pip check
   pip freeze > requirements.lock
   ```
6. Скачайте модели `SG161222/Realistic_Vision_V6.0_B1_noVAE`, `lllyasviel/control_v11p_sd15_canny` и `stabilityai/sd-vae-ft-ema` из Hugging Face и разместите их в каталоге `models/`.

## Запуск
```cmd
python main.py
```
