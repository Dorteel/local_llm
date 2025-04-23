# 🧠 Local LLM + VLM Setup (llama.cpp-based)

Welcome to the pinnacle of DIY AI hubris. This guide sets up large language models (LLMs) and vision-language models (VLMs) locally, with zero cloud nonsense, just raw silicon and ambition.

**System Specs:**
- 🧠 64 GB RAM
- 🎮 NVIDIA GPU (8GB VRAM, CUDA 11.4 — adorable)
- 💨 2GB swap (lol)
- 🐍 Python virtual environment (assumed active)

---

## 📦 Step 1: Clone the Repo and llama.cpp Submodule

```bash
git clone --recurse-submodules https://github.com/Dorteel/local_llm
cd local-llm-vlm
# If you forget the submodule flag:
git submodule update --init --recursive
```

---

## 🧰 Step 2: Install Dependencies

Install the build tools and Python dependencies.

```bash
sudo apt update && sudo apt install -y build-essential cmake python3 python3-pip libopenblas-dev

# Optionally isolate the madness
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio numpy pillow transformers accelerate
```

If you get CUDA errors, don’t cry. Just use CPU. It builds character (and heat).

---

## 🛡️ Step 3: Build `llama.cpp`

```bash
cd llama.cpp
make LLAMA_CUBLAS=1
cd ..
```

If your GPU screams or sobs, it’s working. If it crashes, you're probably using it wrong.

---

## 🧠 Step 4: Download a Model

Pick a quantized GGUF model from Hugging Face. 7B fits your specs. Bigger models will... try.

```bash
mkdir models && cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -O llama-2-7b.Q4_K_M.gguf
cd ..
```

---

## 🤖 Step 5: Run the LLM

```bash
./llama.cpp/main -m models/llama-2-7b.Q4_K_M.gguf -t 8 -n 512
```

- `-t`: threads (you have 32 cores, so let’s not be shy)
- `-n`: token count; higher = slower = more wisdom

---

## 👁️ Step 6: Add a VLM (e.g., LLaVA)

**Install LLaVA dependencies:**

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -r requirements.txt
```

**Download weights and processor configs per LLaVA’s instructions.**

Make sure your GPU doesn’t set itself on fire. LLaVA will try.

---

## 🧪 Step 7: Test LLaVA

```bash
python3 -m llava.serve.cli \
  --model-path ./checkpoints/llava-7b \
  --image-file ./sample.jpg \
  --query "What is happening in this image?"
```

---

## �� Notes

- GPU RAM = tight. Use quantized models or face swap-pocalypse.
- For large models, CPU inference may be your sad but stable friend.
- You can update the repo to include config examples and scripts, but this README is basically already doing all the work.

---

## 🐢 Future Work

- Add web UI for maximum overcompensation.
- Integrate OpenVINO for slightly less pitiful VLM speed.
- Cry when 65B models fail to load, then buy a better GPU.

---


