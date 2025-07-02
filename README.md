
# Renderer

1. Создайте и активируйте виртуальное окружение:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```
2. Обновите `pip` и установите зависимости:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Запуск генерации:
   ```bash
   python main.py --input scetch.png --output result.png --prompt "ultra-realistic minimalistic modern house, evening golden hour lighting, photoreal, 8k, octane render"
   ```