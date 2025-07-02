
# Renderer

This project converts a sketch into a photorealistic render using Stable Diffusion XL and ControlNet.

## Подготовка окружения (Windows 11 + CUDA 12.1)

1. Установите Python 3.10 или новее и создайте виртуальное окружение:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
2. Обновите `pip` и поочередно установите зависимости:
   ```cmd
   python -m pip install --upgrade pip
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 xformers==0.0.31 --extra-index-url https://download.pytorch.org/whl/cu121
   pip install diffusers==0.34.0 transformers==4.45.2 accelerate==0.34.0 peft==0.15.2 huggingface-hub==0.33.1 safetensors==0.4.5
   pip install numpy==1.26.4 opencv-python==4.11.0.86 pillow==10.4.0 tqdm==4.67.0
   ```
3. Скачайте модели `stable-diffusion-xl-base-1.0` и `controlnet-canny-sdxl-1.0` из Hugging Face и разместите их в каталоге `models/`.

## Запуск

```cmd
python main.py --input scetch.png --output result.png --prompt "ultra-realistic minimalistic modern house, evening golden hour lighting, photoreal, 8k, octane render"
```

Скрипт требует доступного CUDA‑GPU. Проверьте, что установлены драйверы NVIDIA и CUDA 12.1.
