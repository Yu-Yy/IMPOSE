# IMPOSE: Identity-Consistent Multi-Pose Generation of Contactless Fingerprints

## 🔍 Overview

**IMPOSE** is a comprehensive generative framework designed to bridge the gap between contact-based and contactless fingerprint modalities. It provides a systematic pipeline for synthesizing high-fidelity, identity-consistent contactless fingerprint datasets under diverse and complex acquisition conditions.

As an essential component of our fingerprint research ecosystem, IMPOSE serves as a powerful tool for data augmentation, significantly enhancing the performance of dense representation frameworks in low-resource or cross-modal scenarios.

The framework integrates three core pillars:

*   🧬 **Identity-Consistent Generation**: Synthesizing master "identities" (rolled fingerprints) using latent diffusion, ensuring intra-class consistency across multiple samples.
*   🌓 **Cross-Modal Texture Synthesis**: Leveraging image-to-image translation to map binary ridge structures into realistic contactless textures with high fidelity.
*   🔄 **Multi-Pose Rendering**: A pose-aware mapping system that projects generated textures onto varying 3D finger orientations to simulate real-world contactless capture.

---

## 🛠 Environment Setup

The environment requirements are identical to **Latent Diffusion Models (LDM)**. Please refer to the [official LDM repository](https://github.com/CompVis/latent-diffusion) for the complete list of dependencies and installation steps.
<!-- ```bash
# Recommendation: Follow LDM's setup, then install additional tools
pip install opencv-python nfiq2
``` -->

## 📂 Model Weights

We provide pre-trained weights for the two primary stages of the IMPOSE pipeline. Please download the checkpoints from the links below and place them into the `models/` directory.

| Model Stage | Function | Link |
| :--- | :--- | :--- |
| **Rolled-Gen** | Master Identity Synthesis | [Download Checkpoint](https://drive.google.com/drive/folders/1p6hCoPb1xrYsKLMbxDxQ6bPqnTmhKeVM?usp=sharing) |
| **Cross-Modal** | Texture & Modality Synthesis | [Download Checkpoint](https://drive.google.com/drive/folders/1CNNgTiC0US60Lc-rSmOHJEKRF6aXxYPp?usp=sharing) |

---

## 📋 Pipeline & Usage

### 1. Rolled Fingerprint Generation (Identity Creation)
Start by generating the base "identities" using the diffusion model.
```bash
python Rolled_FP_generation.py --n_samples 4 --ddim_steps 50 --ddim_eta 0.5 --nums 10
```

### 2. Quality Assurance & Filtering
To ensure dataset quality, generated samples are filtered based on:
*   **NFIQ 2.0 Score** > 0.55
*   **Foreground Ratio** > 60%

### 3. Ridge Enhancement (Sauvola)
Generate binary ridge maps as the structural guidance for texture synthesis.
```bash
python sauvola_simulation_contact.py 
```

### 4. Cross-Modal Texture Generation
Translate the binary structural maps into realistic contactless fingerprint images.
```bash
python cross_modal_generation.py --img-folder example/binary_enhancement --ddim_steps 50
```

### 5. Multi-Pose Texture Mapping
The final step maps the synthetic textures onto various poses to complete the contactless simulation.
```bash
python pose_mapping_generation.py 
```

---

## 🔬 Citation
*(Note: This work is currently under journal review. The citation will be updated upon publication.)*

If you use this code or our generated datasets, please cite our research framework:

```bibtex
@phdthesis{pan2026fingerprint,
  title={Fingerprint Recognition Research Framework for Complex Acquisition Conditions},
  author={Pan, Zhiyu},
  school={Tsinghua University},
  year={2026}
}
```

---

## ⚠️ License

This project is released under the **Academic Research License**. It is provided for academic and educational use only; commercial use is strictly prohibited.

---

## 📬 Contact

For technical issues or collaboration, please reach out to: **Zhiyu Pan** (pzy20@mails.tsinghua.edu.cn)

For research collaboration or other inquiries, please contact the corresponding authors: **Jianjiang Feng** (jfeng@tsinghua.edu.cn)