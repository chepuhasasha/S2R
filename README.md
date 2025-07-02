
# Renderer

Этот проект преобразует эскиз в фотореалистичный рендер с помощью Stable Diffusion XL и ControlNet.

## Подготовка окружения (Windows 11 + CUDA 12.1)

0. Скрипт требует доступного CUDA‑GPU. Проверьте, что установлены драйверы NVIDIA и CUDA 12.8:
   ```cmd
   nvidia-smi
   nvcc --version
   ```

1. Установите Python 3.10 и создайте виртуальное окружение:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
2. Обновите `pip:
   ```cmd
   python -m pip install --upgrade pip
   ```
3. Установите необходимые зависимости:
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
   python -m xformers.info
   ```
5. Проверка и запись версий пакетов:
   ```cmd
   pip check
   pip freeze > requirements.lock
   ```
6. Скачайте модели `stable-diffusion-xl-base-1.0` и `controlnet-canny-sdxl-1.0` из Hugging Face и разместите их в каталоге `models/`.

## Запуск

```cmd
python main.py --input scetch.png --output result.png --prompt "ultra-realistic minimalistic modern house, evening golden hour lighting, photoreal, 8k, octane render"
```