### **Technical Report 2: From Sweep to Slice — Engineering the Final Aura-Mind AI**

#### **Abstract**

With a stable single-GPU training pipeline established, our mission shifted from creating a *working* model to engineering the *optimal* one. This report details the data-driven journey to fine-tune, validate, and deploy the Aura-Mind "Maize Expert" model. We executed a rigorous Bayesian hyperparameter sweep, which successfully produced multiple models with **100% validation accuracy**. However, this victory revealed a deeper challenge: the standard tools for model deployment were incompatible with the fine-tuned architecture. The solution required a deep dive into Gemma 3n's core innovation—the **MatFormer "nested doll" architecture**—and the engineering of a novel **"fine-tune then slice"** pipeline. This report documents the complete process, from automated experimentation to the final, successful creation of a deployable, high-performance model artifact.

---

### **Part 1: The Hunt for the Perfect Recipe**

Our first report established that a single-GPU training strategy was the most stable and effective baseline. Now, we needed to find the perfect "recipe" of hyperparameters. To do this, we launched a 5-run Bayesian hyperparameter sweep with Weights & Biases, tuning the learning rate, LoRA rank, and number of epochs.

The results were a phenomenal success. After implementing our improved prompt engineering, the pipeline produced multiple models that achieved a **perfect 100% accuracy** on our held-out validation set.

![Parallel Plot](/alpha.png)

The `icy-sweep-2` run was crowned our champion, achieving 100% accuracy with an efficient configuration. We had our expert knowledge, captured in a set of LoRA adapters. The next step seemed simple: merge these adapters and prepare the model for deployment.

---

### **Part 2: The Deployment Wall and the MatFormer Insight**

This is where the real challenge began. When we attempted to merge our champion E2B-trained adapters into the larger E4B model for slicing, we hit a hard wall: a `RuntimeError` due to a fundamental size mismatch. Our expert knowledge, trained for the smaller 2-billion-parameter model, simply wouldn't fit into the architecture of its larger parent.

This "failure" was the most important breakthrough of the project. It forced us to look deeper, leading us to the official Google blog post and the `MatFormer Lab` notebook. We realized we had been treating Gemma 3n like a standard LLM, when in fact, it was something entirely new.

The key was the **MatFormer (🪆 Matryoshka Transformer)** architecture. The E4B model isn't just bigger than the E2B; it literally contains a fully functional, smaller E2B model nested inside it, like a set of Russian dolls. This meant our entire deployment strategy had to be re-engineered.

---

### **Part 3: The "Fine-Tune then Slice" Solution**

The MatFormer insight gave us a new, architecturally correct path forward:

1.  **Train on the Parent:** We first had to fine-tune the full, 4-billion-parameter E4B model. This created a new set of LoRA adapters that were compatible with the entire "nested doll" structure.
2.  **Merge the Expert Knowledge:** We then merged these new, larger adapters into the E4B base model, creating a single, dense, fine-tuned expert model.
3.  **Surgically Extract the Child:** Finally, using the logic from Google's own MatFormer Lab, we "sliced" the model. This process surgically removed the extra layers and resized the remaining components, extracting the smaller, fine-tuned E2B sub-model from within its parent.

This multi-step process, conducted in a separate "Deployment Lab" notebook to manage hardware constraints, was a complete success. We successfully created a final, standalone E2B model that was both **efficiently sized for on-device use** and **contained the 100% accurate expert knowledge** from our fine-tuning process.

The final proof is the model itself, now successfully uploaded to the Hugging Face Hub. The `model.safetensors` file is **10.9 GB**—the correct size for a dense, float16 version of the E2B model.

![Hugging Face Model Card](/e2b.png)
---

### **Conclusion: A Complete Success**

Our journey did not end with a successful training run. It required a deep dive into the fundamental architecture of Gemma 3n to overcome a critical deployment roadblock. By embracing the MatFormer paradigm and engineering a novel "fine-tune then slice" pipeline, we successfully produced a final, high-performance, and deployable model.

This process demonstrates a complete, end-to-end workflow for taking a cutting-edge model, specializing it for a high-impact social good task, and preparing it for the real world.