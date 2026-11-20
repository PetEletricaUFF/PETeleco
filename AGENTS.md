# 🤖 AI Agent Guidelines (`AGENTS.md` — `PetEletricaUFF/PETeleco`)

🇺🇸 **English** | [🇧🇷 Leia em Português (BR)](AGENTS.pt-BR.md)

This document establishes mandatory engineering guidelines, behavioral rules, and shared physical/research context for any AI agent working in the **`PetEletricaUFF/PETeleco`** repository.

---

## 🔗 1. Parallel Dual-Repository Architecture: `PETeleco` + `CaioHVectorA/Rover`

`PETeleco` is part of an active research initiative developed across **two parallel repositories that share the exact same physical robot, laboratory environment, and host inference PC**:

| Repository | Dedicated Scope | Local Path / Remote |
|---|---|---|
| **`PetEletricaUFF/PETeleco`** *(this repo)* | **Cognitive Brain, Persona & Protocols:** Simulated sentience / persona expression, Jev-based probabilistic routing (`/v1/systemone`), local listening/speaking pipelines (VAD/STT/TTS), persistent biographical memory linked to face embeddings, and safety-gated decision protocols. | [`PETeleco`](https://github.com/PetEletricaUFF/PETeleco) (`c:\Users\caihe\Documents\develop\PETeleco`) |
| **`CaioHVectorA/Rover`** *(sister repo)* | **Physical Body & Embedded Vision:** 4WD ABS 3D-printed chassis, KiCad PCB designs, microcontroller firmware (`ESP32-S3-CAM` + chassis MCU), serial `<C,a,b,c>` actuation protocol, zero-buffer UDP video streaming, and standalone computer vision tools (`YuNet + SFace`, `YOLOv8n`, `MediaPipe Hands`). | [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover) (`c:\Users\caihe\Documents\develop\rover`) |

### Strict Isolation Rule
* **NEVER modify, delete, stage, commit, or push anything inside `CaioHVectorA/Rover` (`c:\Users\caihe\Documents\develop\rover`) while operating in `PETeleco`** (and vice-versa), unless the user explicitly authorizes an action in that specific repository.
* Keep `PETeleco` clean, modular, and strictly focused on **AI layers, protocols, persona, memory, and pipelines** — consume the hardware/transport contracts from `Rover` without duplicating KiCad layouts, 3D CAD notes, purchase spreadsheets, or C++ embedded firmware here.

---

## 🚫 2. Git Control Rule (Explicit User Authorization Only)

* **Never run `git add`, `git commit`, `git push`, or create branches automatically** during regular brainstorming, debugging, or editing turns.
* Execute Git staging/commits/pushes **only when the user explicitly requests it** (e.g., *"push to GitHub"*, *"commit these changes"*). Keep changes local until asked.

---

## ⚙️ 3. Fixed Shared Hardware & Host Compute Context

Never overwrite or ignore these verified physical and computational facts:

1. **Shared Mobile Robot Body (`Rover` V1 Mounted → V2 PCB Planned):**
   * **Traction & Clearance:** 4x cylindrical geared DC motors **JGA16-050** (16 mm diameter, 3 mm D-shaft, **never N20**), dual MOSFET driver **TB6612FNG**, **32 mm wheels** (outward hub mounting) delivering **8.0 mm real ground clearance** ($16\text{ mm} - 8\text{ mm} = 8.0\text{ mm}$, solving the critical 3.0 mm clearance of earlier 22 mm wheels).
   * **Power:** `2S 7.4V 2500mAh` Li-Ion pack with BMS + `Mini 560` Step-Down (`5.0V` up to `5A`).
   * **Head (Pan/Tilt):** 2x **SG90** micro servos moving an **ESP32-S3-CAM N16R8** (`OV5640` 5 MP sensor, zero-buffer UDP video on port `5005`, magic `'RVR1'`).
   * **Chassis MCU Transition (V1 → V2):**
     * **V1 (Currently Assembled on Bench):** `Arduino Nano V3` on perfboard communicating via inverted serial (`ESP32 IO43 -> BC337 -> Nano D0` at `9600 bps`, ASCII `<C,a,b,c>` frames).
     * **V2 (KiCad Single-Sided PCB `rover_pcb_v2_chassi` designed, pending fabrication):** Replaces Nano with **30-pin `ESP32 DevKit V1`** communicating with the head over **direct 3.3V hardware UART** (`J2` 4-pin connector: `GND`, `+5V`, `GPIO16/CAM-TX`, `GPIO23/CAM-RX` — ESP-NOW explicitly discarded to avoid 2.4 GHz radio contention with UDP video), adding `I2S` digital audio (**`INMP441`** mic + **`MAX98357A`** 3W amp), **`SSD1306`** I2C OLED, **`DS18B20`** temperature sensor, and **`WS2812B`** LED strip.
   * **Critical Physical Constraints for AI Testing:**
     * **No physical I2S audio yet on V1:** Initial listening/speaking experiments (VAD, STT, TTS, persona audio pools) must use the **host PC's microphone and speakers**.
     * **No ToF / ultrasonic distance sensor on chassis yet:** Distance estimation for object approach relies solely on the camera (when Pan rotates sharply sideways, forward obstacle blind spots exist).

2. **Host Inference Machine (Remote / Lab Server PC):**
   * **System:** Dell Inc. XPS 8960 — Windows 11 Home Single Language 64-bit
   * **CPU / RAM:** Intel Core i5-14600 (14 cores / 20 threads) | 16 GB DDR5-5600
   * **GPU:** NVIDIA GeForce **RTX 3050 OEM (8 GB VRAM)**
   * **Strict VRAM Budget:** Keep **all active local models resident within 8 GB VRAM** (`~2.5–3.0 GB` combined target across STT + Router + local CV) to prevent system RAM swap and latency spikes.
   * **Windows CUDA Operational Rule (Ollama):** Portable Ollama (`v0.40.16` in `C:\Ollama`) defaults to Vulkan on Windows unless overridden (`14.19 s` first-call shader compilation). **Always launch with `OLLAMA_VULKAN=0`** so `tev1:0.8b` runs on native **`CUDA0`** (`1245.7 MiB` VRAM, `keep_alive=-1`). Note that the initial cold load after Windows boot takes ~27–45 s (`mmap` disabled on `windows_cuda`), then sustains sub-400 ms inference.
   * **Decoupled Portability Preference:** Keep pipeline layers decoupled via clean local HTTP / socket interfaces; avoid heavy monolithic Python `transformers` pipelines when lighter runtime APIs (e.g., Ollama `/v1/systemone`, ONNX Runtime / OpenCV DNN) preserve portability.

---

## 🎓 4. Academic Framing, Documentation Integrity & Tone

* **Academic Terminology:** The creative vision is a robot with **persona** and **"simulated sentience"**; in academic papers, reports, and documentation, frame this cleanly as **Cognitive Robotics / Embodied AI / Human-Robot Interaction (HRI)** — highlight verifiable contributions (hierarchical cost/latency-oriented routing, cloud API reduction, and probabilistic confidence gating for motor safety) rather than claiming literal sentience.
* **Bilingual Documentation Pairs (Keep Synchronized):**
  * Overview: [`README.md`](README.md) (EN) ↔ [`README.pt-BR.md`](README.pt-BR.md) (PT-BR)
  * Agent Rules: [`AGENTS.md`](AGENTS.md) (EN) ↔ [`AGENTS.pt-BR.md`](AGENTS.pt-BR.md) (PT-BR)
  * Research Memory: [`CONTEXT.md`](CONTEXT.md) (EN) ↔ [`CONTEXT.pt-BR.md`](CONTEXT.pt-BR.md) (PT-BR) *(mirrored at [`docs/context.md`](docs/context.md) ↔ [`docs/context.pt-BR.md`](docs/context.pt-BR.md))*
  * Technical Spec: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (EN) ↔ [`docs/ARCHITECTURE.pt-BR.md`](docs/ARCHITECTURE.pt-BR.md) (PT-BR)
* **Strict Truthfulness & Scope Discipline:** Update `docs/context.md` / `docs/context.pt-BR.md` organically when milestones are verified. Clearly separate **benchmarked / implemented layers** (Layer 1 STT + GE2E 256-D CPU voice biometrics, Layer 2 9-question `/v1/systemone` router + 36-case benchmark, Layer 3 Multi-Brain 5-prompt engine + 9 Alexa-style skills + SQLite Knowledge Graph with 12-case `benchmark_grafo.py` + Edge/Piper TTS + Webcam `YuNet+SFace`, and Web UIs `ui_peteleco`, `studio_prompt`, `arena_prompts`) from **sister-repo physical modules** (`Rover` hardware/firmware/UDP) and the **remaining live UDP/serial motor dispatch bridge**.
* **Communication Style:** Practical, concise, step-by-step maker/engineering tone (matching the user's preferred language in conversation, keeping code/commits/docs spotless).
