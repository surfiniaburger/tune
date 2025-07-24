### **Technical Report: A Comparative Analysis of Training Strategies for the On-Device "Aura-Mind" AI**

#### **Abstract**

To create a viable on-device AI for Nigerian farmers, our project, Aura-Mind, required a fine-tuning process that was not only effective but also efficient and reproducible on accessible hardware. This report details a comparative study between two distinct fine-tuning strategies for our specialized `Gemma 3N` "Maize Expert" model: a stable **single-GPU baseline** and an advanced **multi-GPU distributed approach**. The findings demonstrate that while multi-GPU training offers a path for future scaling, the single-GPU method provided superior convergence and stability, proving to be the optimal strategy for developing a high-quality, production-ready model for our specific use case.

---

### **Introduction: The Need for an Optimal Path**

The goal of Aura-Mind is to deliver a private, offline-first AI diagnostic tool on a basic Android phone. This required fine-tuning the state-of-the-art Gemma 3N model on a custom dataset of Nigerian maize conditions. A key challenge was to achieve this using free, publicly available cloud resources (Kaggle T4 GPUs), mirroring the resource-constrained environment many developers in regions like Nigeria face.

Simply training a model was not enough; we needed to find the *best* way to do it. To this end, we conducted two rigorous experiments to determine the most effective training architecture.

---

### **Experiment 1: Single-GPU Fine-Tuning with Official Unsloth**

This experiment established our baseline for a stable, high-quality training run.

*   **Platform:** Kaggle Notebook (Single GPU T4)
*   **Methodology:** The model was trained for 18 epochs (396 steps) using the official `unsloth` library, a standard `SFTTrainer` configuration, and our custom dataset of 176 images.

#### **Results and Analysis**

![Loss Curve](/training_1gpu.png)

The single-GPU training run was a resounding success, establishing a high benchmark for model performance. The training dynamics, logged via Weights & Biases, tell a clear story of effective learning:

1.  **Training Loss:** The model exhibited a textbook learning curve. The loss began at a high of ~13.0 and, after a rapid "breakthrough" phase around step 20, converged smoothly to a final value of **0.0000**. This indicates that the model achieved complete mastery of the training data.

2.  **Stability and Predictability:** The training process was exceptionally stable, with a predictable linear learning rate decay and a gradient norm that mirrored the loss curve, signifying a healthy and efficient optimization process.

**Conclusion for Experiment 1:** The single-GPU pipeline is a robust, reliable, and highly effective method for producing a specialized, expert model. It serves as our "gold standard" for model quality.

---

### **Experiment 2: Multi-GPU Distributed Training with OpenSloth**

The second experiment explored the potential for accelerated training by leveraging Kaggle's dual T4 GPU instances. This required moving beyond the official library to a community-driven fork (`opensloth`) designed to overcome documented FSDP (Fully Sharded Data Parallel) incompatibilities with the Gemma 3N architecture.

*   **Platform:** Kaggle Notebook (Dual GPU T4 x2)
*   **Methodology:** The model was trained for 18 epochs using the `opensloth` library, launched in a distributed environment via the `accelerate` CLI.

#### **Results and Analysis**

![Loss Curve](/training_2gpu.png)

The multi-GPU experiment successfully overcame the critical `FSDP` and `OOM` errors that plague the official libraries, proving that distributed training is technically feasible.

1.  **Training Loss:** The loss curve demonstrated an even more aggressive initial convergence, dropping from ~13.0 to ~2.0 in fewer than 20 steps. This confirms the theoretical speed advantage of parallel processing. However, the final training loss settled at **~0.62**, which, while excellent, did not reach the near-perfect zero of the single-GPU run.

2.  **Complexity and Overhead:** While successful, this approach introduced significant setup complexity, requiring custom scripts, the `accelerate` launcher, and careful management of the distributed environment.

---

### **Final Comparison and Strategic Decision**

| Metric | Single-GPU (Unsloth) | Multi-GPU (OpenSloth) |
| :--- | :--- | :--- |
| **Final Training Loss** | **~0.0000** (Superior) | ~0.6269 |
| **Setup Complexity** | **Low** (Standard) | **High** (Advanced) |
| **Stability** | **Very High** | Moderate |
| **Primary Advantage** | **Model Quality & Simplicity**| Potential for Speed/Scale |

**Final Verdict:**

While the multi-GPU experiment was a valuable technical exploration that demonstrates a path for future scaling, the **single-GPU training strategy was demonstrably superior for our immediate goal.** It produced a model with a lower final training loss—indicating a higher degree of specialization—within a simpler, more stable, and more easily reproducible workflow.

For the Aura-Mind project, where the ultimate goal is **accuracy and reliability** in the hands of a farmer, the choice was clear. We have proceeded with the model from the single-GPU training run as the foundation for our final Android application. This data-driven decision ensures we are building our product on the most powerful and well-trained AI possible.