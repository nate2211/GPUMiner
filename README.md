<img width="1272" height="697" alt="GPUMiner" src="https://github.com/user-attachments/assets/4f3b1165-78ed-440f-a04b-3c4696735463" />


# ⚡ GPUMiner

A high-performance GPU-focused cryptocurrency miner built with OpenCL, featuring advanced candidate generation, intelligent filtering, and seamless integration with modern mining backends.

Designed for speed, scalability, and experimentation with GPU-accelerated hashing pipelines.

---

## 🚀 Features

### ⚡ GPU Accelerated Mining
- OpenCL-based GPU hashing engine  
- Optimized kernel execution for high throughput  
- Configurable work sizes and tuning parameters  
- Multi-GPU support  

### 🧠 Smart Candidate Generation
- Predictive candidate filtering before verification  
- Tail-based ranking and scoring system  
- Top-K selection for efficient CPU verification  
- Reduced unnecessary hashing overhead  

### 🔗 Backend Compatibility
- Monero RPC integration  
- P2Pool / Stratum compatibility  
- Broker-based job support  
- Flexible job ingestion pipeline  

### 🧮 Advanced Share Processing
- Candidate ranking using:
  - predicted_tail  
  - rank_score  
  - quality metrics  
- Reduced duplicate and low-quality submissions  
- Efficient submit pipeline  

### 📊 Performance Optimization
- Dynamic workload tuning  
- Adaptive scan intensity  
- Queue-aware processing  
- GPU memory optimization  

### 🖥️ Monitoring & Logging
- Real-time GPU stats  
- Hashrate tracking  
- Candidate + share logging  
- Debug + performance logs  

---

## 🏗️ Architecture Overview

```
        ┌────────────────────┐
        │   Job Source       │
        │ (RPC/Stratum/etc) │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Job Manager        │
        └─────────┬──────────┘
                  │
                  ▼
        ┌──────────────────────────┐
        │ GPU Scanner (OpenCL)     │
        │ Kernel Execution         │
        └─────────┬────────────────┘
                  │
                  ▼
        ┌──────────────────────────┐
        │ Candidate Selection      │
        │ (Top-K + Filtering)      │
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │ CPU Verification         │
        │ (RandomX or backend)     │
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │ Share Submission         │
        └──────────────────────────┘
```

---

## 📦 Project Structure

```
gpuminer/
├── gpu_scanner.py         # OpenCL GPU scanning engine
├── opencl_kernels/        # GPU kernels
├── miner_core.py          # Main mining logic
├── job_manager.py         # Job handling
├── verifier.py            # CPU verification
├── submitter.py           # Share submission logic
├── config.py              # Configuration
├── utils/                 # Utilities
└── logs/                  # Runtime logs
```

---

## ⚙️ Installation

### Requirements
- Python 3.10+
- OpenCL-compatible GPU  
- GPU drivers installed  
- (Optional) RandomX runtime for verification  

### Setup

```bash
git clone https://github.com/nate2211/gpuminer.git
cd gpuminer

pip install -r requirements.txt
```

---

## 🔧 Configuration

Example configuration:

```python
GPU_ENABLED = True
GPU_DEVICES = [0]

WORK_SIZE = 256
GLOBAL_WORK_SIZE = 65536

POOL_URL = "stratum+tcp://127.0.0.1:3333"
WALLET_ADDRESS = "YOUR_WALLET"

ENABLE_VERIFICATION = True
```

---

## ▶️ Running the Miner

```bash
python miner_core.py
```

---

## ⚡ GPU Tuning

You can tune performance using:

- `WORK_SIZE` → local workgroup size  
- `GLOBAL_WORK_SIZE` → total threads dispatched  
- Kernel parameters inside `opencl_kernels/`  

Tips:
- Start small, then scale up  
- Monitor GPU temperature and usage  
- Avoid exceeding VRAM limits  

---

## 🧠 Advanced Features

### 🔍 Predictive Filtering
Reduces CPU load by selecting only high-quality candidates before verification.

### 📉 Adaptive Load Control
Adjusts GPU workload dynamically based on system conditions.

### ⚡ Top-K Candidate Selection
Only the best candidates are passed forward, improving efficiency.

### 🧮 Tail-Based Ranking
Uses hash tail analysis to prioritize high-value candidates.

---

## 📊 Performance Notes

- Best performance achieved with tuned OpenCL kernels  
- Works well alongside CPU miners  
- Designed for high-throughput environments  

---

## ⚠️ Disclaimer

This project is for **educational and experimental purposes**.  
Running GPU miners may consume significant power and hardware resources.

---

## 📄 License

MIT License

---

## ⭐ Contributing

Pull requests are welcome.  
For major changes, open an issue first to discuss what you'd like to change.
