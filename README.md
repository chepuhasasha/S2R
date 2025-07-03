# Renderer

Этот проект преобразует эскиз в фотореалистичный рендер с помощью Stable Diffusion XL и ControlNet.
Начиная с этой версии библиотека `xformers` больше не требуется: оптимизацию внимания обеспечивают встроенные функции PyTorch (Flash Attention и SDPA).

## Подготовка окружения (Windows 11, Python 3.10+, CUDA 12.8)

### Требования
1. Visual Studio Build Tools с компонентом "Desktop development with C++"
2. CUDA Toolkit 12.8 и драйверы NVIDIA
3. Установленный Python 3.10, 3.11 или 3.12
4. Активное виртуальное окружение и установленный PyTorch с поддержкой CUDA

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
6. Скачайте модели `stable-diffusion-xl-base-1.0` и `controlnet-canny-sdxl-1.0` из Hugging Face и разместите их в каталоге `models/`.

## Запуск
```cmd
python main.py --input scetch.png --output result.png --prompt "ultra-realistic minimalistic modern house, evening golden hour lighting, photoreal, 8k, octane render"
```
