# 🏗️ Architecture & Protocol Specification (`PETeleco`)

🇺🇸 **English** | [🇧🇷 Leia em Português (BR)](ARCHITECTURE.pt-BR.md)

This document specifies the cognitive pipeline, inter-layer contracts, probabilistic safety gates, persona/memory architecture, and hardware interface protocols of **`PetEletricaUFF/PETeleco`**.

---

## 1. Dual-Repository Architecture: Body (`Rover`) vs. Brain (`PETeleco`)

`PETeleco` and **[`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover)** are developed in parallel on the same physical robot and host PC:

- **`CaioHVectorA/Rover` (Embodied Platform):**
  - 4WD ABS 3D-printed chassis (32 mm wheels, 8.0 mm ground clearance, 4x `JGA16-050` motors, `TB6612FNG` driver).
  - Pan/Tilt head (2x `SG90` servos + `ESP32-S3-CAM OV5640`) streaming zero-delay UDP video.
  - Chassis controller transitioning from **V1** (`Arduino Nano V3` perfboard) to **V2 PCB** (`rover_pcb_v2_chassi` single-sided KiCad board with `ESP32 DevKit V1 30p`, `INMP441` I2S mic, `MAX98357A` I2S amp, `SSD1306` OLED, and `DS18B20` sensor).
  - *Current physical test rule:* Since I2S audio hardware is not yet assembled on V1, initial voice/listening experiments run directly through the host PC microphone and speakers. Also, because there is no ToF/ultrasonic sensor on the chassis yet, object-approach distance estimation relies solely on the camera.
- **`PetEletricaUFF/PETeleco` (Cognitive Core):**
  - Abstracts locomotion/sensing behind clean transport contracts and focuses exclusively on **persona, hierarchical local-first routing, continuous speech interaction, persistent person memory, and safe cloud escalation**.

---

## 2. End-to-End Cognitive & Persona Pipeline

```text
Mic Audio ─▶ Silero VAD ─▶ faster-whisper (CUDA PT-BR) ─┐
                                                         ▼
UDP Camera ─▶ YuNet+SFace / YOLOv8n / Rover State ──▶ ContextoRover.formatar_state()
                                                         │
                                                         ▼
                                            POST /v1/systemone (tev1:0.8b)
                                                         │
        ┌───────────────────┬────────────────────┬───────┴─────────┬──────────────────────┐
        ▼                   ▼                    ▼                 ▼                      ▼
   [ignorar]       [pool_fala_pronta]   [fala_parametrizada]  [acao_rover]       [cognicao_gemini]
   (Noise/TV)       (Canned .wav)        (Deterministic +      (Direct >=0.80 /    (Async Gemini +
   Drop silently     0 ms synthesis       Piper TTS + Cache)   Confirm 0.50..0.79)  Camera Frame?)
```

### Persona & Persistent Memory Layer (Design)
- **Persona Consistency (`0 ms` + Dynamic):** Routine social moments (`saudacao`, `despedida`, `ok_confirmacao`, `elogio_agradecimento`) play pre-recorded persona-voiced `.wav` files so the robot feels lively without TTS latency. Dynamic responses (`fala_parametrizada` and escalated `Gemini` replies) use local **`Piper TTS`** on CPU with persona-styled prompt conditioning.
- **Face-Indexed Biographical Memory (Planned):** Whenever **`YuNet` + `SFace`** (`128-D` embedding, cosine similarity `>= 0.363`) identifies someone in front of the camera, `LOCAL_FACE_ID` enriches the prompt context and enables reading/updating structured facts and conversation summaries in a local **`SQLite` / `JSON`** store keyed by person ID.

---

## 3. Layer 2 Protocol: Ollama `POST /v1/systemone` Contract

Rather than generating free-form JSON text tokens with an LLM, **PETeleco** uses the Jev-based **`/v1/systemone`** endpoint exposed by Ollama (`>= v0.35`, verified on `v0.40.16`) with **`tev1:0.8b`** (`keep_alive=-1`). It evaluates `state` against **7 typed parallel questions** in a single inference pass.

### 3.1 Structured `state` Input (`ContextoRover.formatar_state()`)
```text
USER_SPEECH (pt-BR): "<STT transcript>"
CURRENT_ROVER_MODE: <mapped mode description, e.g. idle_ready (motors off, waiting)>
LOCAL_FACE_ID (SFace): <e.g. Caio (94%) | desconhecido | nenhuma>
LOCAL_OBJECTS_SEEN (YOLO): <e.g. person, bottle>
```

### 3.2 Parallel Questions Schema (`construir_perguntas_rover()`)

| Question ID | Type | Classes / Purpose |
|---|---|---|
| `dirigido_ao_robo` | `noul` | Scalar probability `[0..1]` that speech addresses the robot vs humans/TV |
| `rota_principal` | `choice` | `ignorar`, `pool_fala_pronta`, `fala_parametrizada`, `acao_rover`, `cognicao_gemini` |
| `pool_fala_pronta` | `choice` | `saudacao`, `despedida`, `ok_confirmacao`, `elogio_agradecimento`, `nenhuma` |
| `fala_parametrizada` | `choice` | `hora_atual`, `previsao_clima`, `bateria_temp`, `quem_vejo_agora`, `nenhuma` |
| `acao_rover` | `choice` | `seguir_rosto`, `ir_ate_objeto`, `parar`, `radar`, `nenhuma` |
| `cognicao_gemini` | `choice` | `descrever_cena`, `conversa_aberta`, `nenhuma` |
| `precisa_imagem_camera` | `noul` | Scalar probability `[0..1]` that live camera frame is required |

---

## 4. Probabilistic Safety & Arbitration Engine (`RoteadorSystemOne.decidir()`)

Deterministic Python rules sit on top of `/v1/systemone` calibrated probabilities to prevent unintended motor motion or noisy interruptions:

1. **Emergency Stop Override:**
   - If `acao_choice == "parar"` and `acao_prob >= 0.90` and `p_dirigido >= 0.50` -> immediate `acao_rover:parar` with `requer_confirmacao=False` (even during transient states).
2. **Combined Ambient Chatter / TV Filter:**
   - Returns `ignorar:conversa_paralela_ou_ruido` if any holds (`limiar_dirigido = 0.55`):
     - `p_dirigido < 0.35`
     - `p_dirigido < 0.55 and p_ignorar >= 0.35`
     - `p_ignorar >= 0.65 and p_dirigido < 0.75`
     - `p_ignorar >= 0.55 and p_dirigido < 0.65`
3. **Motor Safety Interlocks (`acao_rover`):**
   - If `acao_choice != "nenhuma"`:
     - `acao_prob >= 0.80` (`limiar_acao_direta`) -> executes directly (`requer_confirmacao=False`).
     - `acao_prob < 0.80` (except `parar`) -> sets `requer_confirmacao=True` so the robot asks for verbal confirmation before energizing motors.
   - If `acao_choice == "nenhuma"` while `rota_prob >= 0.90` and `p_dirigido >= 0.65`, checks the strongest concrete sub-action probability; if `>= 0.25`, escalates safely with `requer_confirmacao=True` instead of silently dropping an imperative phrase; otherwise falls back to non-motor routes to avoid false-positive actuation.
4. **Specialist Deterministic Tie-Breaker:**
   - If `param_choice != "nenhuma"` and `param_prob >= 0.85` while `rota_choice in ("pool_fala_pronta", "cognicao_gemini")` and `p_imagem < 0.55` (`limiar_imagem`), routes to `fala_parametrizada` locally instead of misclassifying telemetry questions as social chat or cloud queries.
5. **Cloud Vision Trigger (`cognicao_gemini`):**
   - Sets `precisa_imagem_camera = True` whenever `p_imagem >= 0.55` or `subtipo == "descrever_cena"`.

---

## 5. Embodied Transport Protocols (Shared with `CaioHVectorA/Rover`)

| Protocol | Transport | Wire Format / Key Details |
|---|---|---|
| **Video & Discovery** | UDP Port `5005` | Magic `0x52565231` (`'RVR1'`), 16-byte `<IHHHHI` header, `HELLO` broadcast discovery, RTT ping (`P` + `<d`), latest-frame socket draining (zero frame queue), resolution commands (`1..9, 0, K, Z`). |
| **Chassis & Head Control** | Serial UART (`<C,a,b,c>` ASCII) | `<P,n>`/`<A,n>` ping-ack, `<S,pan10,tilt10[,vel10]>` adaptive servo speed (0.1° units), `<M,esq,dir>` motors (`-255..255`), `<X>` emergency stop, `<B,mV,minmV>` battery telemetry, `<H,up_s,rx,err>` 1 Hz heartbeat, `<C,id,valor>` runtime tuning. |
| **Cloud Multimodal Client** | HTTPS REST | Asynchronous background thread (`Gemini` API) resizing camera frame to `<= 480 px` JPEG (quality 72) so video/control loops never block. |

---

## 6. Host Compute & Benchmark Reproducibility

- **Target Host:** Dell XPS 8960 | Intel Core i5-14600 | 16 GB DDR5-5600 | **NVIDIA GeForce RTX 3050 OEM (8 GB VRAM)** | Windows 11.
- **Vulkan vs. CUDA Discovery on Windows:**
  - Ollama `v0.40.16` defaults to Vulkan (`OLLAMA_VULKAN=true`), incurring **`14,193.6 ms`** shader compilation on the first call (with preliminary 6-question runs measuring `p50 = 297.9 ms`, `p95 = 321.18 ms`).
  - Launching with **`OLLAMA_VULKAN=0`** forces native **`CUDA0`** (`Flash Attention`), using **`1245.7 MiB`** of VRAM (`/api/ps`). Initial post-boot load on `windows_cuda` takes ~27–45 s because Ollama disables `mmap` on Windows CUDA, after which `keep_alive=-1` keeps inference warm (`min = 306.21 ms`, `mean = 352.85 ms`, **`p50 = 352.54 ms`**, **`p95 = 376.05 ms`**, `max = 491.71 ms`, `n = 72`).
- **36-Case Verified Accuracy (`src/benchmark_router.py --suite todas --rodadas 2`):**
  - **20 standard cases + 16 unseen out-of-sample generalization cases** -> `36/36 (100.0%)` main route, `31/31 (100.0%)` subtype, `34/34 (100.0%)` camera-need detection, and **`30/36 (83.3%)`** resolved locally without cloud calls.

---

## 7. Next Engineering Milestones

1. **Layer 1 Integration:** Local continuous microphone pipeline (`Silero VAD` -> `faster-whisper` CUDA) feeding `ContextoRover`.
2. **Layer 3 Skills & Persona Voice:** Pre-recorded PETeleco `.wav` audio pool, local deterministic handlers (`hora_atual`, `previsao_clima` cache, simulated/real `bateria_temp`, `quem_vejo_agora`), `Piper TTS` synthesis, and `SQLite` person memory.
3. **Layer 4 + Embodied Bridge:** Connect `Gemini` multimodal escalation and dispatch confirmed motor/head states to `CaioHVectorA/Rover` over UDP/serial.
