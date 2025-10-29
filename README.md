# Group-Relative Attention Guidance for Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2501.01234-b31b1b?logo=arxiv&logoColor=white)]([[https://arxiv.org/abs/2501.01234](https://arxiv.org/)](https://arxiv.org/abs/2510.24657))
<!-- [![Project Page](https://img.shields.io/badge/🌐%20Project-Page-blue)](https://yourname.github.io/GRAG-Project)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/your-org/GRAG)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/yourname/GRAG) -->

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

<!-- ### 🚀 Features
- 🎯 **Training-free** — plug-and-play with existing DiT-based pipelines  
- 🪶 **Lightweight** — integrate with as few as *4 lines of code*  
- 🎚 **Continuous Control** — smooth interpolation of editing intensity  
- 🖼 **Compatible Frameworks** — Stable Diffusion 3, FLUX, Qwen-Image, PixArt-α, etc.   -->

<!-- --- -->

<!-- ### 🧩 Installation

```bash
git clone https://github.com/yourname/GRAG.git
cd GRAG
conda create -n grag python=3.10
conda activate grag
pip install -r requirements.txt
```

---

### 🧠 Usage Example

```bash
python run_edit.py   --model flux   --image assets/sample.jpg   --prompt "turn the cloudy sky into a sunset"   --lambda 1.1
```

Results will be saved under `outputs/`.

---

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
- [ ] Release code of grag
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
