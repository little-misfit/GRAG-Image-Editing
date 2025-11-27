# Group-Relative Attention Guidance for Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2510.24657-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.24657)
[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-blue)](https://little-misfit.github.io/GRAG-Image-Editing/)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/Little-ECHO/GRAG-Image-Editing)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/little-misfit/GRAG-Image-Editing)

---

<!-- ### 📖 Paper
> **Group-Relative Attention Guidance (GRAG) for Image Editing**  
> *Your Name, Co-author Name, Advisor Name*  
> *CVPR 2025 (Oral)*  
>
> Recent advances in Diffusion-in-Transformer (DiT) based image editing models have shown powerful editing capabilities, but lack continuous control over the editing strength.  
> We propose **Group Relative Attention Guidance (GRAG)**, a simple yet effective tuning-free mechanism that enables smooth and fine-grained control over editing intensity by reweighting token-wise deltas relative to their shared bias vectors. -->



### 🖼️ Overview

<p align="center">
  <img src="assets/main.jpg" alt="Principle diagram" width="800">
</p>

> GRAG re-weights the delta values between tokens and shared attention biases, enabling fine-grained, continuous editing without training.

---

## 📰 News
- **2025-11-05** — 🎉 [GRAG-ComfyUI](https://github.com/amir84ferdos/ComfyUI-GRAG-ArchAi3D) is now available — 🥳thanks to [Amir Ferdos](https://github.com/amir84ferdos) for his contribution!
- **2025-11-03** — ✨ Added GRAG for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit).

---

### 🧩 Installation

```bash
git clone https://github.com/little-misfit/GRAG-Image-Editing.git
cd GRAG-Image-Editing/Qwen-Edit-GRAG
conda create -n grag python=3.10 -y
conda activate grag
pip install -r requirements.txt
```

---

### 🔧 Usage Example

The recommended range for the grag parameter is **0.8** ~ **1.7**, adjusted in **0.01** increments.

##### One image editing
```bash
sh inference.sh
```

##### Images editing ([PIE](https://github.com/cure-lab/PnPInversion?tab=readme-ov-file) benchmark format)
```bash
sh PIE_test.sh
```

##### Gradio image editing
```bash
python app.py
```

Results will be saved under `results/`.

---

<!-- ### 🚀 Features
- 🎯 **Training-free** — plug-and-play with existing DiT-based pipelines  
- 🪶 **Lightweight** — integrate with as few as *4 lines of code*  
- 🎚 **Continuous Control** — smooth interpolation of editing intensity  
- 🖼 **Compatible Frameworks** — Stable Diffusion 3, FLUX, Qwen-Image, PixArt-α, etc.   -->

<!-- --- -->

<!-- 



### 📂 Repository Structure

```
GRAG/
├── assets/               # figures and demo images
├── src/                  # main code
│   ├── pipeline_grag.py  # GRAG attention modification
│   ├── utils/            # helpers
│   └── eval/             # evaluation scripts
├── demo/                 # web / gradio demo -->
<!-- └── README.md
``` -->

<!-- --- -->

### 🧪 TODO
- [ ] Add huggingface space for grag
- [x] Release code of grag
- [ ] Release visualization code for embedding feature

---

### 💡 Citation

If you find this work useful, please cite:

```bibtex
@misc{zhang2025grouprelativeattentionguidance,
      title={Group Relative Attention Guidance for Image Editing}, 
      author={Xuanpu Zhang and Xuesong Niu and Ruidong Chen and Dan Song and Jianhao Zeng and Penghui Du and Haoxiang Cao and Kai Wu and An-an Liu},
      year={2025},
      eprint={2510.24657},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.24657}, 
}
```

<!-- ---

### 🙌 Acknowledgements
Our implementation builds upon  
[Stable Diffusion 3](https://github.com/Stability-AI/stable-diffusion-3),  
[Flux 1.0](https://huggingface.co/black-forest-labs/FLUX.1-dev),  
and [PixArt-α](https://github.com/PixArt-alpha).

---

### 📧 Contact
For questions or collaborations, please reach out:  
**your.email@example.com** -->
