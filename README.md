### The Spark: A Problem I Couldn't Ignore

This project didn't start in a sterile lab or a corporate boardroom. It started with a memory and a frustration. Growing up in Nigeria, I've always been surrounded by the vibrant, chaotic, and beautiful reality of our local agriculture. I've walked through markets like the one at Boundary, in Apapa, Lagos—a sprawling universe of produce where the hopes of countless farmers are laid out on display. But I've also seen the other side: the quiet desperation in a farmer's eyes when they see their maize leaves yellowing, their tomato plants wilting, their peppers succumbing to a blight they can't name.

This isn't a small problem. Agriculture is the backbone of our economy, employing over [**70% of Nigeria's population**](https://www.google.com/search?q=percentage+of+nigerian+population+in+agriculture), yet it's a sector fighting an uphill battle. We face an estimated [**30-40% in crop losses**](https://www.google.com/search?q=crop+losses+in+nigeria+due+to+pests+and+diseases) annually due to pests and diseases. For a smallholder farmer, that's not just a statistic; it's the difference between sending their children to school and another year of struggle. The knowledge to fight back exists, but it's locked away in academic papers and expert manuals, inaccessible to those who need it most, especially in rural areas where the biggest barrier isn't just the cost of data, but the near-total lack of reliable internet connectivity.

The spark for this project came from a paper by Professor T. O. Anjorin from the University of Abuja, which demonstrated the potential of using deep learning to identify local crop diseases. It was a brilliant proof of concept, but it highlighted a gap: how do we take this powerful lab-based technology and put it directly into the hands of a farmer in a remote village, in a way that works offline, speaks their language, and understands their reality?

**This became my mission: to build a tool that was personal, private, and powerful enough to run in the palm of a farmer's hand.**

### The Proposed Solution: An AI That Lives on the Edge

I envisioned Aura-Mind: an offline-first, AI-powered companion that could run on a basic Android phone. A farmer could simply take a picture of a struggling plant, and the AI would provide a probable diagnosis and simple, actionable, low-cost advice—no internet required.

The launch of Google's Gemma 3N, a powerful multimodal model designed for on-device operation, was the final piece of the puzzle. The technology was finally here. The challenge was to bend it to our will.

---

### The Journey: A Trial by Fire

The path from idea to a working model has been a brutal, beautiful, and deeply personal fight. It has been a microcosm of the very problem we're trying to solve: a battle against constraints, a search for knowledge, and an absolute refusal to quit.

---

#### The First Hurdle: Data, The Real Gold

The academic dataset from Prof. Anjorin's paper was a fantastic starting point, but I quickly realized it wasn't enough. It was clean, academic. I needed the messiness of the real world. I went to the market at Boundary, Apapa, with my phone. I didn't just take pictures of perfect, diseased leaves; I photographed the rejected piles. The bruised tomatoes, the spotted peppers, the maize cobs with their husks half-torn, sitting in wicker baskets under the harsh Lagos sun. I spoke to the sellers, the real experts, and gathered not just images, but context. This project is built on that ground truth.

---

#### The Second Hurdle: The Tools and the "Goliath" of Complexity

My initial plan was to use Apple's MLX framework on my MacBook. It felt like the perfect "David vs. Goliath" story—building a powerful AI on a personal computer. But the libraries were too new, the dependencies too fragile. It was a dead end.

I pivoted to the cloud, using free tiers on Google Colab and Kaggle. This is where the real war began. The next few weeks were a relentless cycle of debugging:
*   We fought **dependency hell**, with `protobuf` and `tensorflow` versions clashing in the dead of night.
*   We battled **cryptic hardware errors**, like the infamous `CUDA:1 and cpu` error on Kaggle's multi-GPU machines, a problem that forced us to understand the deep, internal workings of device mapping.
*   We were mocked by the **"Silent Hang,"** where a training job would run for hours, GPUs blazing at 0%, because of a subtle I/O bottleneck from loading high-resolution images. We solved it by building a robust pre-processing pipeline, only to discover a better way.

---

#### **The Breakthrough: Achieving Perfection, Uncovering a Deeper Truth**

After a relentless war against hardware errors and dependency hell, we finally achieved a stable, single-GPU training pipeline. But a working model wasn't enough. I needed the *best* model.

I unleashed a **Weights & Biases Bayesian Sweep**, an automated agent that would intelligently test five different combinations of hyperparameters on the smaller, efficient **Gemma 3n E2B** model. This was a rigorous, automated tournament to find a champion. To ensure the victory was real, each run was validated against a held-out set of images.

The results were a stunning success. After implementing targeted prompt engineering, the pipeline produced multiple models that achieved **100% accuracy.**

| Run Name          | Learning Rate | LoRA Rank (r) | Epochs | Final `train/loss` | **Validation Accuracy** |
| :---------------- | :------------ | :------------ | :----- | :----------------- | :-------------------- |
| **comic-sweep-1** | **8.34e-06**  | **16**        | **15** | **0.0003**         | **100%**              |
| **stilted-sweep-2** | 1.57e-05      | 32            | 20     | 0.0001             | **100%**              |
| **unique-sweep-3**| 1.16e-05      | 32            | 20     | 0.0001             | **100%**              |
| **vague-sweep-4**   | 1.07e-04      | 32            | 20     | 0.0001             | **100%**              |
| **amber-sweep-5**   | 4.45e-04      | 32            | 20     | 0.0001             | **100%**              |


This was the proof. The data, the prompt, and the pipeline were all correct. I had a collection of perfect models. The next step seemed simple: prepare the champion model for the on-device conversion.

This is where I hit the real wall. And in hitting it, I discovered the true genius of Gemma 3n.

---

#### **The Final Hurdle: The Secret of the Nested Dolls**

My attempts to convert the fine-tuned E2B model failed. The tools, from ONNX to MediaPipe, buckled under the complexity of the new architecture. The reason was deeper than a simple bug. A crucial clue in the official Google documentation revealed the truth: I had been trying to fine-tune one of the smaller "nested dolls" (the E2B model) in isolation.

The true power of Gemma 3n's **MatFormer architecture** required a new, more advanced approach. To create a truly robust and deployable model, I had to work with the entire "Matryoshka" set.

I pivoted my entire deployment strategy to a novel **"fine-tune then slice"** workflow:
1.  **Train the Parent:** I relaunched my sweep, this time fine-tuning the full, 4-billion-parameter **E4B** model. This was a massive undertaking, pushing the limits of Kaggle's hardware and leading to new out-of-memory errors that required a final, aggressive memory management strategy to solve.
2.  **Validate the Expert:** Despite the challenges, the E4B sweep also produced models with **100% validation accuracy**, confirming our approach worked on the larger architecture.
3.  **Surgically Extract the Child:** Finally, using the logic from Google's own MatFormer Lab, I took the winning E4B adapters and surgically "sliced" out the smaller, fine-tuned E2B sub-model.

This multi-stage process, moving from a small model sweep to a large model sweep and then to a final architectural slice, was a complete success. I successfully created a final, standalone E2B model that was both **efficiently sized for on-device use** and **contained the 100% accurate expert knowledge** from our fine-tuning process.

---

## Where We Are Now: A Proven Model at the Final Frontier

Today, the entire journey has culminated in a single, powerful artifact: **`AuraMind-Maize-Expert-E2B-Finetuned`**.

This isn't just a set of adapters; it's a complete, standalone, high-performance model, forged through a multi-stage process of rigorous experimentation and deep architectural engineering. It has been **validated with 100% accuracy** on a real-world dataset and is now publicly available on the Hugging Face Hub, ready for the final deployment step.

The final gauntlet, however, has been the on-device conversion itself. Our mission to use the official Google AI Edge and MediaPipe tools revealed a critical gap between the bleeding-edge Gemma 3n model and the currently available stable tooling. The official `pip install mediapipe` version does not yet recognize the Gemma 3n architecture.

This led to a multi-day odyssey to build the entire MediaPipe framework from source—a trial-by-fire that involved taming a complex C++ build system and resolving deep dependency conflicts with tools like OpenCV. While we successfully built a custom version of the framework, this effort proved that the official conversion path for this brand-new model is not yet stable for external developers.

But this is not a defeat. It is a diagnosis, and a contribution. As part of this project, we have **filed a detailed technical issue with the official MediaPipe team (Issue #6049)**, providing them with a full, reproducible case study to help them add official support for Gemma 3n. Our project has become a real-world stress test, helping to forge a clearer path for all developers who will follow.

My passion to see this through has only been strengthened by this battle. The next steps are crystal clear:

1.  **Engage and Collaborate:** We will continue to work with the Google AI Edge and MediaPipe developers on our GitHub issue, providing any information needed to help create an official, stable conversion path for the entire community.
2.  **Explore All Avenues:** We will not wait. While the official path is being paved, we will leverage the flexibility of our Unsloth pipeline to explore alternative on-device formats like **GGUF**, ensuring we find the fastest path to get this vital tool into the hands of farmers.
3.  **Expand the Mission:** With a proven model and a robust training pipeline, we will begin training new expert models for other critical Nigerian crops like cassava and tomatoes.

The problem is too important, the need is too great, and after this journey, we have a proven, data-driven solution in our hands. We are at the final frontier, and we will find a way through. **Let's go.**

---

### Running the Aura-Mind Application

This project uses a two-part environment to manage conflicting dependencies between the main visual-language model and the text-to-speech (TTS) service. Follow these steps to set up and run the application.

**Prerequisites:**
*   Python 3.11
*   `uv` (or `pip`) for package installation

**Step 1: Set Up the Main Application Environment**

This environment runs the Streamlit UI and the main multimodal model.

```bash
# Create and activate the main virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the required packages
uv pip install -r requirements.txt
```

**Step 2: Set Up the Text-to-Speech (TTS) Service Environment**

This is a separate, isolated environment dedicated to the TTS model.

```bash
# Create and activate the TTS virtual environment inside the tts_service directory
python3.11 -m venv tts_service/.venv_tts
source tts_service/.venv_tts/bin/activate

# Install the TTS requirements
uv pip install -r tts_service/requirements.txt

# Deactivate when done
deactivate
```

**Step 3: Download the Required Models**

The application requires two models: the fine-tuned maize expert model and the Pidgin English TTS model.

1.  **Maize Expert Model:** Ensure the `finetuned_model_for_conversion` directory exists and contains your fine-tuned model files. You can dowload it from Hugging Face (`surfiniaburger/AuraMind-Maize-Expert-E2B-Finetuned`).

2.  **TTS Model:** Download the `orpheus-3b-pidgin-voice-v1` model and place it in the root directory of the project. You can download it from the Hugging Face Hub.

**Step 4: Run the Application**

Once both environments are set up and the models are in place, you can run the Streamlit application.

```bash
# Make sure you are in the main environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run aura.py
```

The application should now open in your web browser. You can upload an image of a maize plant to get a diagnosis and hear the result spoken in Pidgin English.


python run_tts_service.py --text "Hello, this is a test." --model-path ../orpheus-3b-pidgin-voice-v1



docker build -t aura-mind-app .

docker run -p 8501:8501 aura-mind-app


docker builder prune -f && docker build --no-cache -t tune .
