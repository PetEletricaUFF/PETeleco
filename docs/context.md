# 📖 Continuous Technical Memory & Shared Research Context (`context.md` — `PETeleco`)

🇺🇸 **English** | [🇧🇷 Leia em Português (BR)](context.pt-BR.md)

> **Living Engineering Record:** Updated organically whenever design decisions, architecture revisions, physical discoveries, or benchmark metrics are verified. Strictly distinguishes between **verified hardware/software measurements**, **sister-repo embodied components (`CaioHVectorA/Rover`)**, and **planned pipeline stages**.

---

## 1. Project Origin, Persona Vision & Academic Framing

- **Identity & Repository Split (`PetEletricaUFF/PETeleco`):**
  - Developed within **PET Elétrica UFF** (`PetEletricaUFF`), **PETeleco** is directly inspired by and engineered **in parallel** with [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover).
  - Both repositories operate on the **same physical research platform and host inference PC**, cleanly dividing concerns:
    - **`CaioHVectorA/Rover`** owns the physical mechatronics, 3D-printed ABS chassis, KiCad PCB (`rover_pcb_v2_chassi`), microcontroller C++ firmware, zero-delay UDP telemetry/video, and standalone CV utilities.
    - **`PetEletricaUFF/PETeleco`** owns the cognitive software brain: persona expression ("simulated sentience"), Jev-based probabilistic intent routing, continuous voice interaction (VAD/STT/TTS), persistent biographical social memory, and protocol contracts linking perception to motor/head actuation.
- **Creative Goal & Academic Framing:**
  - *Creative vision:* An expressive mobile companion robot with recognizable personality (**persona**), immediate social responses, continuous visual/auditory presence, face recognition (`SFace`), persistent per-person conversation memory, scene understanding, and physical motion execution (`seguir rosto`, `ir ate objeto`, `parar`, `radar`).
  - *Academic framing:* Framed formally as **Cognitive Robotics / Embodied AI / Human-Robot Interaction (HRI)** (avoiding literal "sentience" claims in scientific papers). Its primary research contributions are:
    1. **Cost- and latency-aware hierarchical offloading:** Resolving routine greetings, noise filtering, deterministic status checks, and motion commands locally via SystemOne probability routing (`tev1:0.8b`) + canned audio / CPU `Piper TTS` vs. escalating open/visual queries to cloud VLM/LLM (`Gemini`).
    2. **Probabilistic motor safety interlocking:** Using calibrated choice/noul confidence scores as safety gates before energizing physical motors (`>= 0.80` direct execution, `0.50–0.79` voice confirmation prompt, `>= 0.90` emergency stop override).

---

## 2. Shared Physical Robot Context (`CaioHVectorA/Rover` V1 → PCB V2)

Even though CAD/firmware files remain in `Rover`, PETeleco's AI protocols and constraints directly reflect the real physical hardware on the bench:

### 2.1 Mechanical Chassis, Traction & Power
- **Motors & Drive:** 4x cylindrical geared DC motors **JGA16-050** (16 mm diameter, 3 mm steel D-shaft; **not N20 micromotors**) driven in left/right pairs by a dual MOSFET **TB6612FNG** H-bridge.
- **Wheels & Ground Clearance (Upgraded):** Mounted **32 mm outer-diameter wheels** with outward hubs (all motor housing kept internal). Real ground clearance under the motors:
  $$\text{Wheel Radius (16 mm)} - \text{JGA16-050 Radius (8 mm)} = \mathbf{8.0\text{ mm}}$$
  *(Resolved the critical 3.0 mm clearance limitation of earlier 22 mm StickyMAX wheels.)*
- **Power Subsystem:** `2S 7.4V 2500mAh` Li-Ion battery pack (8.4V full charge) with integrated BMS feeding motor `VM` directly and a **Mini 560 Step-Down** regulator (`5.0V` regulated, up to `5A`) powering electronics and servos.
- **Head Kinematics (Pan/Tilt):** 2x **SG90 9g** micro servos (Pan at base, Tilt on moving bracket carrying the **ESP32-S3-CAM N16R8** directly so the short fragile camera FPC ribbon never twists).

### 2.2 Critical Bench Engineering Diagnostics Resolved on V1
1. **ATmega328P Brown-Out Reset Prevention:** Instant 0–100% PWM startup/reversal across 4 motors caused voltage sag below the ATmega328P `4.3V` BOD threshold. Solved in firmware (`nano_motores.h`) with an 800 ms soft-start PWM ramp and mandatory **150 ms reversal dead-time**, cutting inrush current spikes by over 70%.
2. **Elimination of SG90 "Rebolado" (Pendulum Stick-Slip):** Integer-degree `servo.write()` commands at slow speeds caused ~10–20 µs jumps across plastic bushings. Solved (`nano_servos.h`) with **50 Hz (20 ms) microsecond interpolation** (`544–2400 µs` = 1,856 steps), trapezoidal acceleration, minimum anti-stiction velocity, backlash compensation, and pure Hardware Timer1 Fast PWM mode 14 (`D9/OC1A`, `D10/OC1B`).
3. **OV5640 Thermal Control in Software:** Rather than mounting noisy mechanical fans on the Pan/Tilt head, `OV5640` dissipation is controlled via SCCB register `0x3008 = 0x42` (soft sleep < 1 mA after 3 s idle, waking in ~40 ms via `0x3008 = 0x02`), adaptive `XCLK` (`8–12 MHz`), and FreeRTOS Core 0 isolation — maintaining ~33–35 °C in continuous streaming tests.
4. **Serial Bridge Polarity (`V1`):** `ESP32 IO43 -> BC337 -> Nano D0` NPN inverter is compensated via ESP32 hardware UART2 inversion (`Serial2.begin(9600, SERIAL_8N1, PIN_RX, PIN_TX, true)`), delivering clean non-inverted `5V` idle / `0V` pulse logic to `D0`.

### 2.3 Transition from V1 (Perfboard + Nano) to PCB V2 (`rover_pcb_v2_chassi`)
- **Current Bench Status (V1):** Physically assembled 4WD prototype chassis (3D-printed ABS, perfboard with `ESP32-S3-CAM` + `Arduino Nano V3`). **No further perfboards will be built.**
- **Designed Next Generation (V2 Single-Sided PCB `rover_pcb_v2_chassi`):**
  - **Board Geometry:** `85.0 × 65.0 mm` (~60% area reduction vs. `90 × 150 mm` perfboard), 4x M3 mounting holes at `77 × 57 mm`, single-sided routing + ground plane strictly on `B.Cu` (empty `F.Cu` Gerber) with 6 top-side insulated wire jumpers (`JP1–JP6`) and 4 THT decoupling capacitors (`C1` 470 µF 16V on `+5V`, `C2` 470 µF 25V on `VCC_BAT`, `C3` 100 nF on `VBAT-SENSE`, `C4` 100 nF on `+3V3`). Verified in KiCad 10 with **0 DRC, 0 ERC, and 0 schematic-PCB parity errors** (pending fabrication).
  - **Controller Migration (`Nano V3` → 30-pin `ESP32 DevKit V1`):**
    - **Head ↔ Chassis Link:** Dedicated **3.3V direct hardware UART** over 4-pin connector `J2` (`1=GND`, `2=+5V`, `3=CAM-TX` on `U1.6/RX2/GPIO16`, `4=CAM-RX` on `U1.15/GPIO23`). **ESP-NOW was explicitly rejected** to prevent 2.4 GHz radio contention with continuous high-rate UDP video packets on the `ESP32-S3-CAM`.
    - **Chassis Pinout Summary (`U1`):** `TB6612FNG` (`D32/33/25/26/27/14` + `STBY=D13`, `D12` NC), Servos (`D18/D19`), `I2S` Digital Audio (**`INMP441`** mic + external **`MAX98357A`** 3W amp on `WS=D15, BCLK=D5, DOUT=D17, DIN=D35`), External `I2C` (`SSD1306` OLED `J9` + auxiliary `J10–J12` on `SDA=D21, SCL=D22`), remote `DS18B20` (`D4`), 2S battery ADC (`D34`), and `WS2812B` RGB strip (`D2`).
  - **2nd Purchase Closure & AI Testing Impact:** FlySky receiver, 3W headlight, and thermal pad were permanently cut. `INMP441` + `MAX98357A` remain to be mounted physically with V2. **Therefore, V1 voice/listening tests in `PETeleco` begin using the host PC microphone and speakers**, and because V1/V2 have **no ToF/ultrasonic distance sensor**, forward obstacle proximity still relies strictly on the Pan/Tilt camera.

---

## 3. Embodied Transport & Computer Vision Stack (Validated in `CaioHVectorA/Rover`)

| Subsystem | Module (`Rover`) | Technical Specification & Observed Performance |
|---|---|---|
| **Zero-Delay UDP Video** | `tools/zero_delay_udp.py` / `poc_udp_camera` | UDP **Port `5005`**, magic `0x52565231` (`'RVR1'`), 16-byte `<IHHHHI` header, `HELLO` auto-discovery (self-echo ignored), RTT `P` ping, background socket drainer (`0`-frame queue, **< 45 ms** latency). Measured **22.5 FPS continuous in VGA 640×480** (`2.65–2.77 Mbps`, **0.0% packet drop**), PSRAM DMA support up to **5 MP QSXGA (`2560×1920`)** with micro-paced bursts. |
| **Pan/Tilt Control Law** | `PanTiltTracker` (`zero_delay_udp.py`) | Converts pixel error via `~0.095 °/px` (FOV estimate), commands absolute target with latency dead-time (`wait_until = now + latency + |real|/speed + 40 ms`), IIR low-pass angular velocity filter + conditional lead, adaptive servo speed (`25–130 °/s`). Fixed critical bug on 12/11/2026 where `<S,...>` expects **tenths of degrees** (`90.0°` → `<S,900,900,0>`). |
| **Serial Actuation Frame** | `rover_protocolo.h` | Short ASCII frames `<C,a,b,c>`: `<P,n>`/`<A,n>`, `<S,pan10,tilt10[,vel10]>`, `<M,esq,dir>` (`-255..255`), `<X>` emergency stop, `<B,mV,minmV>`, `<H,up_s,rx,err>`, `<L,estado>`, `<C,id,valor>`. |
| **Biometric Face ID** | `tools/reconhecimento_facial.py` | OpenCV DNN **YuNet** (`face_detection_yunet_2023mar.onnx`, 232 KB; ~4 ms VGA, **16–19 ms** in 5 MP photos via `<=960 px` proxy scale) + `alignCrop()` + **SFace** (`face_recognition_sface_2021dec.onnx`, 38.6 MB, `128-D` float32 embedding, cosine threshold `0.363` against `known_faces.json`). Initializes in XGA (`1024×768`) up to 5 MP. Replaced legacy `rover_radar_vip.py`. |
| **Object Navigation FSM** | `tools/rover_trekking.py` | **YOLOv8n** ONNX with aspect-ratio `_letterbox(640, 640)` (avoided 33% distortion on vertical 480×640 camera orientation), target grouping (`bottle`, `cup`, `wine glass`, `vase`), calibrated thresholds (`0.28` acquire / `0.18` hold), live selectors `[A]`, `[O]`, `[D]`. 6-state FSM: `STANDBY` → `RADAR_SCAN` → `TARGET_LOCKED` → `ALIGN` → `APPROACH` → `ORBIT`. |
| **Hand Gestures** | `tools/rover_vision_suite.py` | Migrated to MediaPipe Tasks **`HandLandmarker`** (`hand_landmarker.task`, 7.8 MB, 21 3D landmarks) because `mediapipe 1.2.3` dropped `mp.solutions`. Tuned with `700 ms` hand-loss timeout and `4.5s` continuous-command ceiling. |
| **Async Cloud Co-Pilot** | `GeminiVisionClient` | Background worker calling **`gemini-3.5-flash`**, resizing JPEG frame (`<=480 px` qual `72` in shared utility; `320×240` `<12 KB` payload, `<100 ms` upload, `15s` timeout in Trekking) without stalling video or motor loops. |

---

## 4. Host Inference Machine & 5-Layer Cognitive Architecture (`PETeleco`)

- **Host Server Specs:** Dell Inc. XPS 8960 | Windows 11 Home 64-bit | Intel Core i5-14600 (14C/20T) | 16 GB DDR5-5600 | **NVIDIA GeForce RTX 3050 OEM (8 GB VRAM)** | Python `3.12.10` | `torch 2.6.0+cu124`.
- **Resident VRAM Budget Target (`<= 8 GB`, aiming ~2.5–3.0 GB total):**
  - Layer 1 (Planned): `Silero VAD` (CPU) + `faster-whisper` (`base` or `small`, CUDA `~0.5–1.0 GB` VRAM).
  - Layer 2 (Implemented): Ollama `/v1/systemone` + **`tev1:0.8b`** (`772.85M`, `Q8_0`, `qwen35`, **`1245.7 MiB` VRAM** resident).
  - Layer 3 (Hybrid): `Piper TTS` (CPU `0 GB` VRAM) + canned `.wav` persona clips (`0 ms` TTS) + OpenCV ONNX vision (`YuNet`, `SFace`, `YOLOv8n`) + `SQLite` / `JSON` per-face biographical memory.
  - Layer 4 (On-Demand): `Gemini` multimodal REST client (triggered only when Layer 2 selects `cognicao_gemini`).

---

## 5. Chronological AI Log & Benchmark Progression (`PETeleco`)

### 14/11/2026 — Infrastructure Setup, Vulkan vs. CUDA Diagnosis, Router Calibration & Repo Creation

1. **Portable Runtime Setup:**
   - Installed portable **Ollama v0.40.16** to `C:\Ollama` (added to Windows User `PATH`) and pulled **`tev1:0.8b`** (`784.52 MiB` GGUF Q8_0, `772.85M` params).
   - Verified `winget`, `choco`, `scoop`, `tailscale`, `cloudflared`, and `ngrok` are absent on the host PC; `curl.exe` and `ssh.exe` are present.
2. **Backend Discovery — Windows Vulkan vs. CUDA:**
   - Initial `ollama serve` defaulted to `OLLAMA_VULKAN=true`, spending **`14,193.6 ms`** compiling Vulkan shaders on the first call (warm preliminary 6-question runs measured `min=276.34 ms`, `mean=301.58 ms`, `p50=297.9 ms`, `p95=321.18 ms`, `max=444.45 ms`, `10/20 (50.0%)` route accuracy before prompt tuning).
   - Killing Vulkan and launching with **`OLLAMA_VULKAN=0`** switched inference to native **`CUDA0`** (`ggml-cuda.dll`, `Flash Attention enabled`, `USE_GRAPHS=1`, `keep_alive=-1`).
   - Windows CUDA loads `tev1:0.8b` in ~27–45 s on cold start after daemon boot because Ollama disables `mmap` on `windows_cuda`, consuming **`1245.7 MiB`** in `/api/ps` (`+1333 MiB` `nvidia-smi` delta observed at initial cold load). Once loaded, calls remain consistently warm (~353 ms).
3. **Iterative Layer 2 Router Calibration (`src/router_systemone.py`):**
   - Built decoupled `RoteadorSystemOne` (`POST /v1/systemone`) and 36-case PT-BR benchmark (`20` standard + `16` unseen out-of-sample generalization phrases with zero keyword leakage):
     - *Baseline CUDA (v1 prompts, 20 cases):* `p50=270.29 ms`, `p95=283.71 ms`, Route `10/20 (50.0%)`.
     - *v3 (20 cases):* `p50=354.94 ms`, Route `20/20 (100.0%)`, Subtype `17/17 (100.0%)`, Image `19/19 (100.0%)`.
     - *v4 (36 cases, stripped keyword hints):* `p50=339.74 ms`, Route `26/36 (72.2%)`, Image `33/34 (97.1%)` — exposed unsafe `else "radar"` fallback when `acao_rover` returned `nenhuma`.
     - *v5 (36 cases, removed `else "radar"`):* Route `17/36 (47.2%)`, Noise `5/5 (100%)` — `dirigido_ao_robo` criteria over-penalized normal questions.
     - *v6 (36 cases):* `p50=335.5 ms`, `p95=372.43 ms`, Route `28/36 (77.8%)`, Subtype `23/31 (74.2%)`, Image `34/34 (100.0%)` — root-caused overlap between `pool_fala_pronta` and factual questions in `rota_principal`, plus narrow `seguimento` / `elogio` phrasing.
     - ***v7 Final Verified (`--suite todas --rodadas 2`, `n=72` warm inferences):***
       - Cold-start (resident): `496.1 ms`
       - Warm latency: `min = 306.21 ms` | `mean = 352.85 ms` | **`p50 = 352.54 ms`** | **`p95 = 376.05 ms`** | `max = 491.71 ms`
       - Route accuracy: **`36/36 (100.0%)`** (`20/20` standard + `16/16` out-of-sample generalization)
       - Subtype accuracy: **`31/31 (100.0%)`** *(5 noise cases have no subtype)*
       - Camera-need accuracy: **`34/34 (100.0%)`**
       - Cloud API savings (resolved locally): **`30/36 (83.3%)`**
       - Safety interlock verification: `acao_seguir` (`0.55`) and `acao_ir_objeto` (`0.50`) triggered `[CONFIRMAR]` (`< 0.80`), while explicit `parar` (`0.98`), `radar` (`0.84`), and unseen `gen_acao_1..4` (`0.87–1.00`) executed directly.
   - *Strict Scope Disclaimer:* Benchmarked on isolated PT-BR text + simulated sensor context only (no live microphone/STT, TTS, or physical chassis motors attached yet).
4. **Remote Port Exposure Assessment:**
   - Evaluated remote testing options while operating via Remote Control: exposing raw Ollama port `11434` is unsafe (no auth). Authenticated HTTPS tunnels (requiring installing `cloudflared` or `tailscale`) work well for lightweight text/JSON Jev router evaluation, whereas remote browser microphone streaming for Layer 1 STT distorts latency measurements and triggers browser HTTPS/driver friction — audio/hardware loops are best tested locally on the host PC.
5. **Creation of `PetEletricaUFF/PETeleco`:**
   - Dedicated public GitHub repository initialized and pushed (`main`, keeping `CaioHVectorA/Rover` untouched and evolving in parallel) with bilingual documentation (`EN` + `PT-BR`) and modular Layer 2 Python package (`src/router_systemone.py`, `src/benchmark_router.py`).
6. **Full Interactive Workbench, Offline `Piper` TTS & Linux CPU Portability (2026-11-14):**
   - Implemented and verified end-to-end on the personal Linux laptop (**AMD Ryzen 7 5700U / 30 GB RAM / Ollama `v0.40.16` CPU**, no NVIDIA GPU), keeping full compatibility with **Windows / RTX 3050 CUDA**:
     - **Ollama CPU vs. Renoir Vulkan on Linux:** Confirmed `OLLAMA_IGPU_ENABLE=1` (Vulkan `AMD RADV RENOIR`, `compute=0.0`) hangs indefinitely after `load_tensors`; clean CPU execution completed `--suite padrao --rodadas 1` (`n=20`) with `cold = 8868.1 ms`, warm `min = 6341.74 ms` | `mean = 9217.45 ms` | **`p50 = 9734.7 ms`** | **`p95 = 10678.57 ms`** | `max = 12256.82 ms`, preserving Route **`20/20 (100.0%)`**, Subtype **`17/17 (100.0%)`**, and Image **`19/19 (100.0%)`**.
     - **Adaptive Audio & STT (`src/escuta_stt.py`):** Automatic sample-rate negotiation & resampling to `16 kHz` when hardware rejects native `16000 Hz`, speech-start callbacks for the HUD, automatic Whisper sizing (`small` on CUDA / `base` on CPU — warm transcription `~1.1–2.3 s` vs `~3.1 s` for `small`), and CPU `torchaudio_192d` voiceprint extraction (`~83 ms`).
     - **Cross-Platform Offline TTS (`src/tts_local.py`):** Automatic download and CPU synthesis via **`Piper TTS`** (`pt_BR-faber-medium`, `~262 ms` for ~4.3 s speech) over `sounddevice` / `paplay` / `pw-play` / `aplay`, with Windows SAPI and `espeak-ng`/`spd-say` fallbacks.
     - **Live Webcam + HUD (`src/visao_webcam.py` & `src/testar_peteleco.py`):** Thread-safe persistent `WebcamLive` capture running `YuNet + SFace` (`~10.8 ms`/frame), optional HUD preview (`--preview` / `/preview`), quick webcam face enrollment (`/rosto <name>`), hands-free continuous loop (`--continuo` / `/continuo`), and `.env.example` documenting optional keys (`GEMINI_API_KEY`).
     - **Local Web UI ("Under the Hood", `src/ui_peteleco.py` + `src/static/ui_peteleco.html`):** Dependency-free `ThreadingHTTPServer` streaming MJPEG (`/video.mjpg`) with `YuNet + SFace` bounding boxes, 5 facial keypoints, and cosine scores, paired with an SSE telemetry panel (`/api/events`) showing live pipeline stages, `/v1/systemone` probability bars, hot JSON + cold SQLite memory, table-edge safety locks, mic/TTS/enrollment controls, and optional local Ollama auto-start (`python -m src.ui_peteleco`).
7. **Live Voice, Wake-Word Active Listening & Routing Ergonomics Upgrade (2026-11-14):**
   - Addressed real-room mic/speaker feedback across 6 layers without breaking offline RTX 3050 CUDA operation:
     - **Natural PT-BR Neural Voice (`src/tts_local.py`):** Switched default TTS mode (`auto`) to `edge-tts` (`pt-BR-AntonioNeural`, plus selectable `Thalita`/`Francisca` voices and `rate="+4%"`), decoded in-memory via PyAV (`~3.54 s` synthesis verified for 6.73 s spoken audio), keeping `gemini-2.5-flash-preview-tts` optional (`--tts-modo gemini`) and calmer offline `Piper TTS` (`length_scale=1.14`, `~2.80 s` for 3.41 s audio) as fallback.
     - **Higher-Accuracy PT-BR STT (`src/escuta_stt.py`):** Replaced CPU `base` (`1026 ms` mishearing `"Que dia é hoje?"` as `"Kijia e Ouji."`) with `--stt-modo auto`, using multimodal `gemini-3.5-flash-lite` audio REST on CPU-only PCs (`1493–2567 ms`, 3/3 exact transcriptions on PT-BR test phrases) and `faster-whisper` (`small`, `beam_size=5`, `vad_filter=False`, 450 ms pre-roll VAD, `limiar_rms=0.012`) as CUDA primary / offline CPU fallback.
     - **Standby ↔ Active Listening (`"PETeleco"` Wake-Word):** Continuous mode starts in Standby ignoring third-party chatter until `"PETeleco"` opens a rolling **60 s Active Listening** window (`EstadoEscutaAtiva`) where `ignorar` noise drops are disabled (`exigir_wake_continuo`, UI countdown & manual toggle), automatically closing on farewell or timeout.
     - **CPU Speed + Intent Fixes (`src/router_systemone.py`):** Added `--roteador auto` (`Gemini Flash-Lite` JSON in `~692–1910 ms` on CPU vs `p50 = 9.97 s` Ollama CPU on Ryzen 7 5700U), fixed directional commands (`"Vamos para a frente, pai."` -> `acao_rover`), open joke prompts (`-> cognicao_gemini`), and phonetic date queries (`"Que de hoje?"` -> `fala_parametrizada:hora_atual` speaking weekday + full date). After tightening Jev `ignorar` thresholds (`2026-11-14`), 36-case `benchmark_router.py` regression on CPU `tev1:0.8b` scored **35/36 route (97.2%)**, **30/31 subtype (96.8%)**, and **34/34 image (100.0%)** (`local_clima` misclassified to `hora_atual` and `ruido_terceiros_1` escalated to `cognicao_gemini`).
     - **Conversational Rhythm & Visitor Mode (`prompts/persona_peteleco.md`, `src/persona_peteleco.py`):** Removed gratuitous harsh swearing examples, injected live `DATA_HORA_ATUAL`, converted `"kkkk"` / `"haha"` to speakable pauses for TTS, and fixed `eh_modo_formal()` so unknown visitors stay warm/playful (`NORMAL`) while Prof. Felipe Sass and `SERIO` mode stay strictly formal.

---

## 6. Roadmap (Next Steps)

1. **Live Microphone & Speaker Validation on Real Room Audio:** Validate pre-roll VAD (`0.012`) + `"PETeleco"` wake-word + `Edge Neural TTS` playback over real PC mic/speakers (`python -m src.ui_peteleco`).
2. **Optional `Silero VAD` Upgrade for Noisy Labs:** Evaluate supplementing the RMS pre-roll gate in `src/escuta_stt.py` with `Silero VAD` if multi-person lab chatter falsely wakes Standby.
3. **Embodied `Rover` Bridge (UDP + Serial `<C,a,b,c>`):** Dispatch safety-gated router decisions and surface interlocks over the live `Rover` UDP stream (`port 5005`) and chassis firmware (`Rover` V1 -> V2 `I2S`/`OLED` once fabricated).
