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

Rather than generating free-form JSON text tokens with an LLM, **PETeleco** uses the Jev-based **`/v1/systemone`** endpoint exposed by Ollama (`>= v0.35`, verified on `v0.40.16`) with **`tev1:0.8b`** (`keep_alive=-1`). It evaluates `state` against **9 typed parallel questions** (7 core routing/safety questions + dynamic Multi-Brain & Modular Skills routing) in a single inference pass.

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
| `cerebro_persona` | `choice` | Dynamic Jev routing across specialized brain prompts (`resenha`, `desgracado`, `tecnico`, `serio`, `utilitario`) |
| `skill_modular` | `choice` | Dynamic Jev routing across 9 pluggable skills (`clima`, `musica`, `noticias`, `recados`, `timer`, `busca_web`, `sonoplastia`, `listas`, `rotinas`, `nenhuma`) |

### 3.3 Modular Multi-Brain & Pluggable Skills Layer (`src/cerebros.py`, `src/skills/`)
- **Multi-Brain Prompts (`prompts/cerebros/*.md`):** Instead of a single monolithic prompt with conflicting mode rules, each persona version (`resenha`, `desgracado`, `tecnico`, `serio`, `utilitario`) is a standalone `.md` file with frontmatter (`temperature`, `max_tokens`, `criterio_jev`, `bloqueado_para_professor`) and real `user`/`model` dialogue examples (`=== FEW_SHOT_DIALOGO ===`). `GerenciadorCerebros.resolver_cerebro()` enforces hard deterministic safety locks (e.g., `professor_orientador` and `SERIO` ALWAYS lock to `serio`) on top of Jev's probabilistic `cerebro_persona` distribution, reducing prompt dilution and cutting input tokens per turn.
- **9 Alexa-Style Modular Skills (`src/skills/`):** Pluggable skills (`clima` via Open-Meteo Niterói, headless ad-free `musica`, `noticias` via RSS, `recados` in SQLite with automatic face/voice delivery, background `timer`, free `busca_web`, `sonoplastia` with `.wav` SFX + anti-cringe cooldown, `listas` persisted in graph memory, and `rotinas` `"bom_dia"`/`"modo_estudo"`) dynamically register their `criterio_jev` into `RoteadorSystemOne` and execute deterministically (`0 ms` LLM) or inject verified `DADOS_SKILL` into `utilitario` / `serio`.

### 3.4 Embedded Knowledge Graph Memory (`src/grafo_memoria.py`, `src/benchmark_grafo.py`)
- **`GrafoMemoriaSQLite` (`data/memoria_longa.sqlite`):** Stores normalized entities (`grafo_nos`) and typed directional relations (`grafo_arestas`: e.g. `estuda`, `cursa`, `mora_em`, `trabalha_em`, `usa`, `gosta_de`, `evita`, `lista_item`) using `contextlib.closing` for lock-safe SQLite I/O.
- **Token-Light Subgraph Retrieval:** Before calling Gemini, `buscar_subgrafo()` retrieves a **1–2 hop neighborhood** around the recognized speaker (`[MEMÓRIA EM GRAFO RELEVANTE]`), injecting structured facts without inflating system prompts.
- **12-Case Benchmark (`python -u -m src.benchmark_grafo --modo todos`):**

| Extraction Mode | Precision | Recall | F1 | Latency / Case | Extra Tokens |
|---|---|---|---|---|---|
| **`local`** (regex rules + normalization) | **`1.000`** | `0.889` | **`0.941`** | **`0.18 ms`** | **`0`** (`R$ 0`) |
| **`hibrido`** (`local` + LLM gate on narrative turns) | `0.900` | **`1.000`** | **`0.947`** | `810.57 ms` | `293` |
| **`llm`** (free-form JSON via `gemini-3.5-flash-lite`) | `0.545` | `0.667` | `0.600` | `9099.92 ms` | `2167` |

### 3.5 Neural CPU Voice Biometrics (`src/voz_biometria.py`) & Headless Audio Ducking (`src/skills/musica.py`)
- **256-D GE2E Speaker ID (`0 MB` VRAM):** Pure-PyTorch `_GE2EEncoder` (`LSTM(40, 256, 3)` + `Linear(256, 256)` on linear power MelSpectrogram `n_fft=400, hop_length=160, n_mels=40`) loads `models/voice_encoder_ge2e.pt` (`17.1 MB`) on CPU (`torchaudio_192d` kept as backward-compatible fallback). `identificar_detalhado()` computes `estimar_snr_db()` and applies a **2-tier confidence gate** (`>= 0.78` -> `"confirmado"`, `0.65..0.77` -> `"confirmar"` triggering conversational verification in `persona_peteleco.py`, `< 0.65` -> `"desconhecido"`).
- **Ad-Free Headless Music & Automatic Ducking:** `PlayerMusicaHeadless` resolves queue metadata via `ytmusicapi` (auto-radio), audio streams via `yt-dlp` `scsearch1` (SoundCloud M4A/MP3) and direct `SomaFM` Icecast stations (`lofi`, `eletronica`, `rock`, `ambiente`), decoding and streaming directly through `PyAV + sounddevice` without launching a browser. Hooks in `SintetizadorTTS.falar()` and `EscutaSTT.gravar_microfone()` call `set_music_ducking(True/False)` to smoothly drop playback volume to **`18%`** while listening or speaking.

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
  - **20 standard cases + 16 unseen out-of-sample generalization cases (7 core questions):** `36/36 (100.0%)` main route, `31/31 (100.0%)` subtype, `34/34 (100.0%)` camera-need detection, and `30/36 (83.3%)` resolved locally without cloud calls.
  - **9-question Multi-Brain + Skills schema (`--rodadas 1`):** Route **`34/36 (94.4%)`**, Subtype **`30/31 (96.8%)`**, Image **`34/34 (100.0%)`**, **`p50 = 388.49 ms`**, VRAM **`1245.7 MiB`** (`+0 MiB` delta).

### 6.1 Complete Hardware & Cloud Quota Footprint Projection

| Subsystem | VRAM (`RTX 3050 8 GB`) | System RAM | CPU / Inference Latency | Disk / Cloud Quota |
|---|---|---|---|---|
| **L1 — VAD + `faster-whisper small`** | `~1050–1200 MiB` (`float16` CUDA) / `0 MiB` (`Gemini Audio`) | `~350 MB` | `~350–650 ms` warm CUDA | `~485 MB` |
| **L1 — Voice Biometrics (`ge2e_256d` CPU)** | **`0 MiB`** | `~45 MB` | `~25–45 ms` / utterance | `17.1 MB` (`voice_encoder_ge2e.pt`) |
| **L2 — Jev Router `tev1:0.8b` (`CUDA0`)** | **`1245.7 MiB`** (`keep_alive=-1`) | `~400 MB` | **`p50 = 388.49 ms`** (9 parallel questions) | `~810 MB` (`C:\Ollama`) |
| **L3 — Vision (`YuNet + SFace` OpenCV CPU)** | **`0 MiB`** | `~60 MB` | **`~10.8 ms`** / frame | `~37 MB` |
| **L3 — Multi-Brain + SQLite Graph + 9 Skills** | **`0 MiB`** | `< 25 MB` | **`0.18 ms`** local extract + `< 1.5 ms` 2-hop query | `< 5 MB` (`~5–10 MB/yr`) |
| **L3 — Headless Music (`PyAV`) + TTS (`Edge`/`Piper`)** | **`0 MiB`** | `~80 MB` | `< 2%` of 1 core (music) / `~260–400 ms` (Piper) | `~60 MB` (`Piper` + `.wav` SFX) |
| **L4 — Cloud API (`Gemini 3.5 Flash-Lite` Free Tier)** | `0 MiB` | `~0 MB` | On-demand REST (`0` calls on `~78–83%` of turns) | `~1050–1320` in / `~20–85` out tok/call |
| **Rover V1 → V2 PCB Power (`Mini 560 5V/5A`)** | — | — | MCU/Telemetry only (AI offloaded to PC) | `~180–260 mA` CAM + V2 `~136 mA` logic + `~150–300 mA` avg `MAX98357A` (`~600 mA` 3W peak) |

- **Total Active PC Resident Usage:** **`~2.3–2.5 GB` VRAM** (`~30%` of RTX 3050 8 GB, leaving ~5.5 GB free) and **`~1.1–1.4 GB` System RAM** (`< 9%` of 16 GB DDR5), using **`~1.4 GB`** local disk.

---

## 7. Next Engineering Milestones

1. **Acoustic Room Validation:** Stress-test real lab room noise against pre-roll VAD, 2-tier `ge2e_256d` speaker checks, and `"PETeleco"` wake-word + `18%` headless music ducking via `python -m src.ui_peteleco`.
2. **Optional `Silero VAD` Upgrade for Crowded Labs:** Evaluate supplementing the RMS pre-roll gate in `src/escuta_stt.py` with CPU `Silero VAD` if multi-person background chatter triggers unnecessary Standby wakeups.
3. **Layer 4 + Embodied `Rover` Bridge (UDP + Serial `<C,a,b,c>`):** Dispatch safety-gated router actions and physical table-edge states directly to `CaioHVectorA/Rover` (`port 5005` UDP + V1/V2 chassis UART, plus `I2S`/`OLED` on V2 once fabricated).
