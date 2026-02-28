
# SkyNova: Mastering Multi-Task Remote Sensing with Mixture of LoRA Experts

#### Heng Zhou , Quanjun Zhang , Junjie Huang , Jingzhou Chenand Liang Xiao
\* Equally contributing first authors

#### **the School of Computer Science and Engineering, Nanjing University of Science and Technology**

[![paper]()
[![hf_dataset]()
[![hf_model]()

---

## SkyNova: Overview

Current remote sensing vision-language models are mostly derived by fine-tuning general-purpose VLMs. However, due to the low proportion and limited coverage of remote sensing expertise in general models’ pre-training corpora, these models underperform on tasks requiring specialized knowledge. To address this, we first propose RSCBQA, the first closed-book question answering benchmark tailored for remote sensing. Its training set injects structured domain knowledge to enhance models’ grasp of core concepts, while the test set evaluates factual knowledge mastery. We further propose SkyNova, a multi-task RS-VLM framework based on Mixture-of-LoRA-Experts achitecture, enabling efficient multi-task adaptation without sacrificing domain knowledge. It freezes the general VLM backbone to retain foundational knowledge and adopts lightweight, plug-and-play LoRA experts for parameter-efficient adaptation. To boost expert diversity and mitigate expert collapse, we design a Top-k entropy balancing loss, which maximizes the entropy of Top-k expert activation weights to promote balanced expert participation. Notably, we establish a four-stage progressive training strategy, including domain knowledge injection, task-supervised fine-tuning, preference optimization, and expert refinement, to guide the model toward professional, factually consistent remote sensing comprehension. We also release SkyCorpus, a 10-million-sample large-scale multimodal remote sensing instruction dataset, supporting full-process instruction tuning and domain knowledge internalization. Extensive experiments show SkyNova achieves state-of-the-art performance on most remote sensing benchmarks, with significant advantages over existing RS-VLMs on RSCBQA.

---
## Contents
- [Install](#install)
- [Model Weights and Demo]()
- [Dataset]()

## Install

1. Clone this repository and navigate to SkyNova folder
```bash
git clone https://github.com/zhouheng/SkyNova.git
cd SkyNova
```

2. Install Package
```Shell
conda create -n SkyNova python=3.9 -y
conda activate SkyNova
pip install --upgrade pip  
pip install -e .
```

3. Install additional packages for training cases
```
pip install flash-attn==2.3.6 --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip uninstall transformers
pip install -e .
```

##  Weights and Demo

The SkyNova models are available on the Hugging Face Hub.

[SkyNova_4B]()

---
Please check out our Model for all public checkpoints, and check demo section.
```bash
bash demo.sh
```
---
## Dataset
Coming soon 
---

## 🙏 Acknowledgement
the School of Computer Science and Engineering, Nanjing University of Science and Technology for their collaborative support and guidance.


