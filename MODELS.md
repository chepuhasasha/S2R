# Список моделей и требования к видеопамяти

Ниже приведены несколько распространённых моделей Stable Diffusion и ControlNet, которые можно использовать в данном проекте. Указаны ориентировочные требования к памяти при FP16 и ссылка на страницу модели в Hugging Face.

| Модель | Требуемая VRAM | Ссылка на Hugging Face |
|-------|---------------|------------------------|
| **Stable Diffusion v1.5** | ~4 ГБ | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| **Stable Diffusion 2.1** | ~5 ГБ | [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| **SD-Turbo** | ~2 ГБ | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) |
| **DreamShaper / Realistic Vision / др.** | ~4 ГБ | Примеры: [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper), [SG161222/Realistic_Vision_V6.0](https://huggingface.co/SG161222/Realistic_Vision_V6.0) |
| **ControlNet Canny для SD 1.5** | ~0.8 ГБ | [lllyasviel/control_v11p_sd15_canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) |
| **ControlNet Canny для SD 2.1** | ~1 ГБ | [lllyasviel/control_v11p_sd21_canny](https://huggingface.co/lllyasviel/control_v11p_sd21_canny) |
| **ControlNet Canny для SDXL** | ~2.1 ГБ | [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) |

Точные значения потребляемой памяти могут различаться в зависимости от выбранных опций (offload на CPU, attention slicing, размер изображения и т.д.), однако таблица позволяет прикинуть, поместится ли модель в доступную VRAM.
