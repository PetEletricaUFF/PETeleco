# 📖 Memória Técnica Contínua & Contexto Compartilhado de Pesquisa (`context.pt-BR.md` — `PETeleco`)

[🇺🇸 Read in English](context.md) | 🇧🇷 **Português (BR)**

> **Registro Vivo de Engenharia:** Atualizado organicamente a cada decisão arquitetural, descoberta física de bancada ou benchmark validado. Distingue rigorosamente **medições reais comprovadas**, **subsistemas do repositório irmão (`CaioHVectorA/Rover`)** e **etapas planejadas**.

---

## 1. Origem do Projeto, Visão de Persona & Enquadramento Acadêmico

- **Identidade & Divisão em Dois Repositórios (`PetEletricaUFF/PETeleco`):**
  - Desenvolvido no âmbito do **PET Elétrica UFF** (`PetEletricaUFF`), o **`PETeleco`** nasce diretamente inspirado e evoluindo **em paralelo** com o projeto [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover).
  - Ambos os repositórios operam sobre **exatamente o mesmo robô físico e o mesmo PC servidor**, separando responsabilidades com clareza:
    - **`CaioHVectorA/Rover`** concentra engenharia mecânica (chassi 3D ABS), eletrônica/PCB KiCad (`rover_pcb_v2_chassi`), firmwares C++ (`ESP32-S3-CAM` + controlador do chassi), vídeo/telemetria UDP zero-buffer e scripts isolados de visão computacional.
    - **`PetEletricaUFF/PETeleco`** concentra o cérebro cognitivo: expressão de **persona** ("simulação de senciência"), roteamento probabilístico Jev-based (`/v1/systemone`), pipeline conversacional de áudio (VAD/STT/TTS), memória biográfica persistente por pessoa e protocolos de segurança decisória.
- **Objetivo Criativo & Enquadramento Científico:**
  - *Proposta criativa:* Um robô móvel expressivo com personalidade marcante (**persona**), respostas sociais imediatas, presença audiovisual contínua, reconhecimento biométrico facial (`SFace`), memória persistente de conversas anteriores por indivíduo, descrição de cenas e atuação física segura (`seguir rosto`, `ir até objeto`, `parar`, `radar`).
  - *Enquadramento acadêmico:* Formalizado como **Robótica Cognitiva / Embodied AI / Interação Humano-Robô (HRI)** — evitando afirmar "senciência literal" em textos científicos e destacando os diferenciais mensuráveis:
    1. **Arquitetura hierárquica orientada a custo e latência:** Resolver triagem de ambiente, respostas sociais, consultas determinísticas e comandos físicos localmente (`tev1:0.8b` + `.wav` + `Piper TTS`) antes de pagar custo/latência de LLM/VLM em nuvem (`Gemini`).
    2. **Travas probabilísticas para segurança motora:** Exigir confiança calibrada antes de energizar motores (`>= 0,80` execução direta; `0,50–0,79` confirmação verbal; parada explícita `>= 0,90` prioritária).

---

## 2. Contexto Físico Compartilhado do Robô (`CaioHVectorA/Rover` V1 → PCB V2)

Mesmo mantendo CAD e firmwares no `Rover`, toda decisão de IA e protocolo no `PETeleco` obedece às limitações e conquistas reais da bancada física:

### 2.1 Mecânica, Tração & Alimentação
- **Motores & Ponte H:** 4x motores DC cilíndricos com caixa de redução metálica **JGA16-050** (16 mm de diâmetro, eixo D de aço de 3 mm — **NÃO são N20**) acionados em pares esquerdo/direito pelo driver MOSFET duplo **TB6612FNG**.
- **Rodas & Vão Livre Real (Atualizado):** Montagem com **rodas de 32 mm** e cubo voltado para fora (caixa de redução 100% interna ao chassi). Vão livre ao solo:
  $$\text{Raio da Roda (16 mm)} - \text{Raio do JGA16-050 (8 mm)} = \mathbf{8{,}0\text{ mm}}$$
  *(Superou o vão livre crítico de 3,0 mm das antigas rodas StickyMAX de 22 mm, permitindo rodagem em pisos reais).*
- **Energia:** Pack Li-Ion `2S 7.4V 2500mAh` (8,4V plena carga) com BMS integrado alimentando `VM` dos motores e regulador Step-Down **Mini 560** (`5,0V` fixos / até `5A`) para eletrônica e servos.
- **Cabeça Pan/Tilt:** 2x micro servos **SG90 9g** (Pan na base e Tilt no braço superior carregando a placa **ESP32-S3-CAM N16R8**, protegendo o cabo flat curto/frágil da câmera `OV5640`).

### 2.2 Diagnósticos Físicos Críticos Já Resolvidos na V1
1. **Prevenção de Brown-Out Reset no Nano:** Partidas/reversões instantâneas (0→100% PWM) causavam afundamento (*sag*) abaixo do limiar BOD de `4,3V` do ATmega328P. Resolvido em firmware (`nano_motores.h`) com rampa progressiva de 800 ms e **tempo morto obrigatório de 150 ms na reversão**, reduzindo picos em > 70%.
2. **Eliminação do "Rebolado" nos Servos SG90:** Comandos lentos em graus inteiros sofriam *stick-slip* na bucha plástica. Resolvido (`nano_servos.h`) com interpolação em **microssegundos a 50 Hz (20 ms)** (`544–2400 µs` = 1.856 passos), rampa trapezoidal, piso anti-atrito, compensação de folga e **Hardware Timer1 Fast PWM modo 14** (`D9/OC1A`, `D10/OC1B`).
3. **Controle Térmico por Software do OV5640:** Em vez de ventoinha mecânica na cabeça, usa soneca no registrador SCCB `0x3008 = 0x42` (< 1 mA após 3s sem uso, despertando em ~40 ms via `0x3008 = 0x02`), `XCLK` adaptativo (`8–12 MHz`) e isolamento no Core 0 do FreeRTOS — mantendo ~33–35 °C em transmissão contínua.
4. **Inversão Serial V1 (`IO43 -> BC337 -> Nano D0`):** Compensada nativamente ativando inversão de hardware na UART2 do ESP32 (`Serial2.begin(9600, SERIAL_8N1, PIN_RX, PIN_TX, true)`), entregando repouso em `5V` e pulsos limpos ao pino `D0` do Nano.

### 2.3 Transição da V1 (Perfurada + Nano) para a PCB V2 (`rover_pcb_v2_chassi`)
- **Status Físico Atual (V1):** Protótipo 4WD montado em chassi 3D ABS e placa perfurada funcional (`ESP32-S3-CAM` + `Arduino Nano V3`). **Nenhuma nova placa perfurada será feita.**
- **Próxima Geração Projetada (PCB V2 Face Simples `rover_pcb_v2_chassi`):**
  - **Placa no KiCad 10:** `85,0 × 65,0 mm` (~60% menor que `90 × 150 mm`), 4 furos M3 a `77 × 57 mm`, trilhas e plano de terra 100% em `B.Cu` (Gerber `F.Cu` vazio), 6 jumpers isolados pelo topo (`JP1–JP6`) e 4 capacitores THT (`C1` 470 µF 16V em `+5V`, `C2` 470 µF 25V em `VCC_BAT`, `C3` 100 nF em `VBAT-SENSE`, `C4` 100 nF em `+3V3`). **Validada em CAD com 0 erros de DRC, 0 de ERC e 0 problemas de paridade** (aguardando fabricação).
  - **Migração de MCU (`Nano V3` → `ESP32 DevKit V1` 30 pinos):**
    - **Enlace Cabeça ↔ Chassi:** **UART direta em 3.3V** no conector de 4 vias `J2` (`1=GND`, `2=+5V`, `3=CAM-TX` em `U1.6/RX2/GPIO16`, `4=CAM-RX` em `U1.15/GPIO23`). **ESP-NOW foi descartado** para evitar disputa na antena 2.4 GHz durante o envio intenso de pacotes UDP da câmera.
    - **Pinagem no Chassi (`U1`):** `TB6612FNG` (`D32/33/25/26/27/14` + `STBY=D13`, `D12` NC), Servos (`D18/D19`), Áudio Digital `I2S` (**`INMP441`** mic + **`MAX98357A`** amp 3W externo em `WS=D15, BCLK=D5, DOUT=D17, DIN=D35`), `I2C` externo (`SSD1306` OLED `J9` + canais auxiliares `J10–J12` em `SDA=D21, SCL=D22`), `DS18B20` remoto (`D4`), leitura de bateria (`D34`) e barra `WS2812B` (`D2`).
  - **Fechamento da 2ª Compra & Impacto nos Testes de IA:** Cortados em definitivo receptor FlySky (`FS-iA6B`), farol LED 3W e thermal pad; `INMP441` e `MAX98357A` aguardam montagem na V2. **Por isso, os testes iniciais de fala/escuta do `PETeleco` usam microfone e saída de áudio do PC host**, e como não há sensor ToF/ultrassom no chassi ainda, a estimativa de distância depende exclusivamente da câmera Pan/Tilt.

---

## 3. Stack de Protocolos & Visão Computacional (Validado no `CaioHVectorA/Rover`)

| Subsistema | Módulo (`Rover`) | Especificação Técnica & Desempenho Observado |
|---|---|---|
| **Vídeo UDP Zero-Buffer** | `tools/zero_delay_udp.py` / `poc_udp_camera` | UDP **Porta `5005`**, magic `0x52565231` (`'RVR1'`), cabeçalho de 16 bytes `<IHHHHI`, descoberta `HELLO` (eco próprio ignorado), ping RTT (`P`), thread dedicada drenando socket (`fila zero`, **< 45 ms** latência). Medidos **22,5 FPS contínuos em VGA 640×480** (`2,65–2,77 Mbps`, **0,0% de perda**), com buffers DMA em PSRAM até **5 MP QSXGA (`2560×1920`)** e micro-pacing UDP. |
| **Controle Pan/Tilt** | `PanTiltTracker` (`zero_delay_udp.py`) | Converte erro em px (`~0,095 °/px`), envia alvo absoluto com tempo morto (`wait_until = now + latency + |real|/speed + 40 ms`), filtro IIR de velocidade + avanço preditivo condicional e velocidade adaptativa (`25–130 °/s`). Correção crítica em 12/11/2026: protocolo espera **décimos de grau** (`90,0°` → `<S,900,900,0>`). |
| **Protocolo Serial ASCII** | `rover_protocolo.h` | Quadros curtos `<C,a,b,c>`: `<P,n>`/`<A,n>`, `<S,pan10,tilt10[,vel10]>`, `<M,esq,dir>` (`-255..255`), `<X>` parada de emergência, `<B,mV,minmV>`, `<H,up_s,rx,err>`, `<L,estado>`, `<C,id,valor>`. |
| **Biometria Facial** | `tools/reconhecimento_facial.py` | OpenCV DNN **YuNet** (`face_detection_yunet_2023mar.onnx`, 232 KB; ~4 ms VGA, **16–19 ms** em fotos 5 MP reduzindo cópia `<=960 px`) + `alignCrop()` + **SFace** (`face_recognition_sface_2021dec.onnx`, 38,6 MB, embedding `128-D`, limiar cosseno `0,363` contra `known_faces.json`). Inicia em XGA (`1024×768`) até 5 MP. Unificou o antigo `rover_radar_vip.py`. |
| **FSM Trekking & Objetos** | `tools/rover_trekking.py` | **YOLOv8n** ONNX com `_letterbox(640, 640)` (corrigiu esticamento de 33% em câmera vertical 480×640), agregação (`bottle`, `cup`, `wine glass`, `vase`), limiares `0,28` (aquisição) / `0,18` (retenção), teclas `[A]`, `[O]`, `[D]`. Máquina de 6 estados: `STANDBY` → `RADAR_SCAN` → `TARGET_LOCKED` → `ALIGN` → `APPROACH` → `ORBIT`. |
| **Gestos da Mão** | `tools/rover_vision_suite.py` | Migrado para MediaPipe Tasks **`HandLandmarker`** (`hand_landmarker.task`, 7,8 MB, 21 marcos 3D) após remoção de `mp.solutions` no `mediapipe 1.2.3`. Calibrado com timeout de `700 ms` e teto contínuo de `4,5 s`. |
| **Co-Piloto Multimodal** | `GeminiVisionClient` | Background thread com **`gemini-3.5-flash`**, frame JPEG redimensionado (`<=480 px` qual `72` no utilitário comum; `320×240` `<12 KB`, upload `<100 ms`, timeout `15 s` no Trekking) sem travar vídeo ou motores. |

---

## 4. Máquina Hospedeira & Arquitetura Cognitiva em 5 Camadas (`PETeleco`)

- **PC Hospedeiro:** Dell Inc. XPS 8960 | Windows 11 Home 64-bit | Intel Core i5-14600 (14C/20T) | 16 GB DDR5-5600 | **NVIDIA GeForce RTX 3050 OEM (8 GB VRAM)** | Python `3.12.10` | `torch 2.6.0+cu124`.
- **Meta de Orçamento de VRAM (`<= 8 GB`, alvo combinado ~2,5–3,0 GB):**
  - Camada 1 (Planejada): `Silero VAD` (CPU) + `faster-whisper` (`base` ou `small`, CUDA `~0,5–1,0 GB` VRAM).
  - Camada 2 (Implementada): Ollama `/v1/systemone` + **`tev1:0.8b`** (`772.85M`, `Q8_0`, `qwen35`, **`1245,7 MiB` VRAM** residentes).
  - Camada 3 (Híbrida): `Piper TTS` (CPU `0 GB` VRAM) + clipes `.wav` da persona (`0 ms` TTS) + Visão OpenCV ONNX (`YuNet`, `SFace`, `YOLOv8n`) + memória biográfica `SQLite` / `JSON` por pessoa.
  - Camada 4 (Sob Demanda): Cliente REST assíncrono `Gemini` (acionado apenas quando a C2 escala `cognicao_gemini`).

---

## 5. Registro Cronológico de IA & Evolução do Benchmark (`PETeleco`)

### 14/11/2026 — Infraestrutura, Diagnóstico Vulkan vs. CUDA, Calibração do Roteador e Criação do Repositório

1. **Instalação Portátil do Ollama:**
   - Instalado **Ollama v0.40.16** portátil em `C:\Ollama` (adicionado ao `PATH` do usuário) e baixado **`tev1:0.8b`** (`784,52 MiB` GGUF `Q8_0`, `772.85M` parâmetros).
   - Verificado que `winget`, `choco`, `scoop`, `tailscale`, `cloudflared` e `ngrok` não estão instalados neste PC; `curl.exe` e `ssh.exe` estão disponíveis.
2. **Descoberta Crítica — Vulkan vs. CUDA no Windows:**
   - Sem variáveis extras, o `ollama serve` no Windows iniciava em `OLLAMA_VULKAN=true`, gastando **`14.193,6 ms`** compilando shaders na 1ª chamada (execuções preliminares de 6 perguntas em Vulkan mediram `min=276,34 ms`, `média=301,58 ms`, `p50=297,9 ms`, `p95=321,18 ms`, `max=444,45 ms`, acurácia inicial `10/20 (50,0%)`).
   - Reiniciando com **`OLLAMA_VULKAN=0`**, o modelo passou para **`CUDA0`** nativo (`ggml-cuda.dll`, `Flash Attention enabled`, `USE_GRAPHS=1`, `keep_alive=-1`).
   - A 1ª carga fria pós-boot em Windows CUDA leva ~27–45 s porque o Ollama desativa `mmap` em `windows_cuda`, alocando **`1245,7 MiB`** no `/api/ps` (`+1333 MiB` observados no `nvidia-smi` na carga inicial). Depois de residente, opera continuamente abaixo de 400 ms.
3. **Evolução Iterativa do Roteador (`src/router_systemone.py`):**
   - Criado cliente desacoplado `RoteadorSystemOne` (`POST /v1/systemone`) e bateria de **36 frases PT-BR** (`20` padrão + `16` inéditas out-of-sample sem vazamento de palavras-chave dos critérios):
     - *Baseline CUDA (v1, 20 casos):* `p50=270,29 ms`, `p95=283,71 ms`, Rota `10/20 (50,0%)`.
     - *v3 (20 casos):* `p50=354,94 ms`, Rota `20/20 (100,0%)`, Subtipo `17/17 (100,0%)`, Imagem `19/19 (100,0%)`.
     - *v4 (36 casos, sem dicas nos casos):* `p50=339,74 ms`, Rota `26/36 (72,2%)`, Imagem `33/34 (97,1%)` — expôs bug do fallback `else "radar"` quando `acao_rover` retornava `nenhuma`.
     - *v5 (36 casos, remoção do fallback `radar`):* Rota `17/36 (47,2%)`, Ruído `5/5 (100%)` — `dirigido_ao_robo` penalizava perguntas diretas normais.
     - *v6 (36 casos):* `p50=335,5 ms`, `p95=372,43 ms`, Rota `28/36 (77,8%)`, Subtipo `23/31 (74,2%)`, Imagem `34/34 (100,0%)` — diagnosticada sobreposição entre `pool_fala_pronta` e perguntas factuais em `rota_principal`, além de critérios curtos para `seguir_rosto`/`elogio`.
     - ***v7 Final Validado (`--suite todas --rodadas 2`, `n=72` chamadas warm):***
       - Cold-start (modelo residente): `496,1 ms`
       - Latência warm: `min = 306,21 ms` | `média = 352,85 ms` | **`p50 = 352,54 ms`** | **`p95 = 376,05 ms`** | `max = 491,71 ms`
       - Acurácia de Rota Principal: **`36/36 (100,0%)`** (`20/20` padrão + `16/16` inéditas)
       - Acurácia de Subtipo: **`31/31 (100,0%)`** *(excluindo os 5 casos de ruído sem subtipo)*
       - Acurácia de Câmera (`precisa_imagem_camera`): **`34/34 (100,0%)`**
       - Resolução Local (economia da API Gemini): **`30/36 (83,3%)`**
       - Travas de segurança em ação: `acao_seguir` (`0,55`) e `acao_ir_objeto` (`0,50`) acionaram `[CONFIRMAR]` (`< 0,80`), enquanto `parar` (`0,98`), `radar` (`0,84`) e `gen_acao_1..4` (`0,87–1,00`) passaram diretamente.
   - *Escopo explícito:* Testado apenas sobre texto PT-BR + contexto simulado dos sensores (ainda sem microfone/STT, TTS ou chassi conectado).
4. **Análise de Exposição Remota de Portas:**
   - Concluído que abrir a porta `11434` do Ollama diretamente é inseguro (sem autenticação). Túneis HTTPS autenticados (baixando `cloudflared` ou `tailscale`) funcionam bem para testes leves de texto no Roteador Jev, mas áudio em tempo real (STT/microfone) remoto sofre com RTT de rede, HTTPS `getUserMedia` e drivers, sendo melhor validado presencialmente no PC.
5. **Criação do Repositório `PetEletricaUFF/PETeleco`:**
   - Inicializado e publicado em paralelo ao `CaioHVectorA/Rover` com documentação bilíngue (EN / PT-BR) e código inicial da Camada 2 (`src/router_systemone.py` e `src/benchmark_router.py`).
6. **Bancada Interativa Completa, TTS Offline (`Piper`) & Portabilidade Linux CPU (14/11/2026):**
   - Implementados e validados no notebook de desenvolvimento (**Linux / AMD Ryzen 7 5700U / 30 GB RAM / Ollama `v0.40.16` CPU**, sem GPU NVIDIA), preservando 100% de compatibilidade com **Windows / RTX 3050 CUDA**:
     - **Ollama CPU vs. Vulkan Renoir em Linux:** Confirmado que `OLLAMA_IGPU_ENABLE=1` (Vulkan `AMD RADV RENOIR`, `compute=0.0`) trava indefinidamente após `load_tensors`; execução limpa em CPU concluiu `--suite padrao --rodadas 1` (`n=20`) com `cold = 8868,1 ms`, warm `min = 6341,74 ms` | `média = 9217,45 ms` | **`p50 = 9734,7 ms`** | **`p95 = 10678,57 ms`** | `max = 12256,82 ms`, mantendo Rota **`20/20 (100,0%)`**, Subtipo **`17/17 (100,0%)`** e Imagem **`19/19 (100,0%)`**.
     - **Áudio & STT adaptativo (`src/escuta_stt.py`):** Reamostragem automática para `16 kHz` caso o dispositivo de captura não suporte `16000 Hz` nativo, callbacks para HUD (`on_speech_start`), perfil automático de Whisper (`small` em CUDA / `base` em CPU — transcrição warm `~1,1–2,3 s` em CPU e `~3,1 s` com `small`) e fallback `torchaudio_192d` na CPU (`~83 ms`).
     - **TTS Offline Cross-Platform (`src/tts_local.py`):** Download automático e síntese offline em CPU com **`Piper TTS`** (`pt_BR-faber-medium`, `~262 ms` por ~4,3 s de fala) via `sounddevice` / `paplay` / `pw-play` / `aplay`, com fallbacks para SAPI (Windows) e `espeak-ng`/`spd-say`.
     - **Webcam ao Vivo + HUD (`src/visao_webcam.py` & `src/testar_peteleco.py`):** Captura persistente thread-safe `WebcamLive` com detecção `YuNet + SFace` (`~10,8 ms`/frame), janela HUD opcional (`--preview` / `/preview`), cadastro rápido de rosto (`/rosto <nome>`) e modo contínuo mãos-livres (`--continuo` / `/continuo`) com `.env.example` documentando variáveis opcionais (`GEMINI_API_KEY`).
     - **UI Web Local ("Debaixo dos Panos", `src/ui_peteleco.py` + `src/static/ui_peteleco.html`):** Servidor `ThreadingHTTPServer` sem dependências externas exibindo stream MJPEG (`/video.mjpg`) com caixas faciais `YuNet + SFace`, 5 keypoints e score cosseno, além de painel SSE ao vivo (`/api/events`) com estágios da pipeline, distribuição de probabilidades do `/v1/systemone`, ficha quente/SQLite, trava de bancada, controles de microfone/TTS/cadastro e inicialização automática opcional do Ollama local (`python -m src.ui_peteleco`).
7. **Evolução de Voz Neural, Escuta Ativa por Wake-Word e Roteamento Rápido (14/11/2026):**
   - Ajustes baseados em testes reais com microfone e caixa de som mantendo plena compatibilidade offline com a RTX 3050 (CUDA):
     - **Voz Natural em PT-BR (`src/tts_local.py`):** Modo padrão (`auto`) migrado para **`edge-tts`** (`pt-BR-AntonioNeural`, com seletor para `Thalita`/`Francisca` e `rate="+4%"`), decodificado em memória via PyAV (`~3,54 s` de síntese observada para 6,73 s de fala), mantendo `gemini-2.5-flash-preview-tts` opcional (`--tts-modo gemini`) e **`Piper TTS`** offline mais calmo (`length_scale=1.14`, `~2,80 s` para 3,41 s de áudio) como fallback.
     - **STT de Alta Precisão PT-BR (`src/escuta_stt.py`):** Substituído Whisper `base` em CPU (`1026 ms`, erro *"Que dia é hoje?"* -> *"Kijia e Ouji."*) por `--stt-modo auto`, priorizando áudio multimodal `gemini-3.5-flash-lite` em máquinas sem GPU (`1493–2567 ms`, 3/3 transcrições exatas nos testes PT-BR) e `faster-whisper small` (`beam_size=5`, `vad_filter=False`, VAD com 450 ms pre-roll, `limiar_rms=0,012`) na CUDA ou fallback offline.
     - **Standby ↔ Escuta Ativa (Wake-Word `"PETeleco"`):** O modo contínuo inicia em *Standby* ignorando conversas paralelas até detectar `"PETeleco"`, abrindo janela rolante de **60 s em Escuta Ativa** (`EstadoEscutaAtiva`) sem descartar turnos por `ignorar` (com badge/contador regressivo e controles na UI Web), fechando automaticamente em despedidas ou timeout.
     - **Latência em CPU & Correções de Intenção (`src/router_systemone.py`):** Adicionado `--roteador auto` (`Gemini Flash-Lite` JSON em `~692–1910 ms` na CPU vs `p50 = 9,97 s` Ollama CPU no Ryzen 7 5700U), além de regras para movimento livre (*"Vamos para a frente, pai."* -> `acao_rover`), pedidos abertos de piada (`-> cognicao_gemini`) e fallback fonético (*"Que de hoje?"* -> `fala_parametrizada:hora_atual` respondendo dia da semana + data completa). Após calibrar o limiar `ignorar` do Jev (`14/11/2026`), a suíte de regressão de 36 casos em CPU (`tev1:0.8b`) obteve **35/36 acurácia de rota (97,2%)**, **30/31 subtipo (96,8%)** e **34/34 imagem (100,0%)** (`local_clima` caiu em `hora_atual` e `ruido_terceiros_1` escalou para `cognicao_gemini`).
     - **Ritmo de Bate-Papo e Modo Visitante (`prompts/persona_peteleco.md`, `src/persona_peteleco.py`):** Removidos exemplos com palavrão gratuito/ácido, injetada `DATA_HORA_ATUAL` no contexto dinâmico do LLM, convertidos `"kkkk"` / `"haha"` para pausas naturais de TTS e corrigido `eh_modo_formal()` para visitantes (`desconhecido`) responderem no modo `NORMAL` leve e bem-humorado sem quebrar a formalidade estrita com o Prof. Felipe Sass ou `--modo SERIO`.

---

## 6. Próximos Passos Planejados

1. **Validação Presencial com Microfone & Caixa de Som:** Testar em bancada o VAD com pre-roll (`0,012`), wake-word `"PETeleco"` e reprodução `Edge Neural TTS` via `python -m src.ui_peteleco`.
2. **Silero VAD Opcional em Ambiente Ruidoso:** Avaliar reforço do gate RMS atual em `src/escuta_stt.py` pelo `Silero VAD` caso conversas paralelas no PET acordem o Standby indevidamente.
3. **Ponte Física com `Rover` (UDP + Serial `<C,a,b,c>`):** Conectar decisões aprovadas pelo roteador e travas corporais ao fluxo UDP (`porta 5005`) e comandos do chassi (`Rover` V1 -> V2 `I2S`/`OLED` quando fabricada).
