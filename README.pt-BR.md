# 🤖 PETeleco — Robótica Cognitiva, Persona & Pipeline de IA em Camadas

[🇺🇸 Read in English](README.md) | 🇧🇷 **Português (BR)**

**PETeleco** é o repositório dedicado à arquitetura de IA cognitiva, construção de persona e protocolos de comunicação desenvolvido no **PET Elétrica UFF** ([`PetEletricaUFF`](https://github.com/PetEletricaUFF)). O projeto implementa um agente incorporado expressivo com simulação de senciência (academicamente enquadrado como **Robótica Cognitiva / Embodied AI / Interação Humano-Robô — HRI**), capaz de ouvir, conversar em Português (BR), reconhecer pessoas, manter memória biográfica por rosto, descrever cenas e acionar o corpo físico com segurança.

---

## 🔗 Inspiração & Desenvolvimento Paralelo com `CaioHVectorA/Rover`

O `PETeleco` é diretamente inspirado e desenvolvido em paralelo com o projeto **[`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover)**:

| Repositório | Papel no Sistema Compartilhado | Escopo |
|---|---|---|
| **[`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover)** | **Corpo Físico & Visão Embarcada** | Chassi 4WD em ABS, PCB KiCad, `ESP32-S3-CAM` (`OV5640`), controlador do chassi (`Nano V3` → `ESP32 DevKit`), vídeo UDP, protocolo serial `<C,a,b,c>` e scripts de visão (`YuNet + SFace`, `YOLOv8n`, `MediaPipe`). |
| **`PetEletricaUFF/PETeleco`** *(este repo)* | **Cérebro Cognitivo, Persona & Protocolos** | Arquitetura hierárquica local-first, roteamento probabilístico Jev-based, pipeline de voz (VAD/STT/TTS), memória persistente indexada por rosto, persona e travas probabilísticas de segurança. |

Os dois repositórios evoluem em paralelo sobre o **mesmo contexto físico e hardware de pesquisa**, mantendo o repositório de IA simples e desacoplado de desenhos CAD, esquemáticos e firmwares C++.

---

## 💡 A Ideia: Persona Incorporada Orientada a Custo e Latência

Mandar toda frase capturada pelo microfone e todo frame de câmera para um LLM/VLM na nuvem torna o robô lento, caro, vulnerável a conversas paralelas no ambiente e arriscado para movimentar motores diretamente.

O **PETeleco** resolve isso com uma **arquitetura hierárquica em 5 camadas**:
1. **Triagem local & Wake-word imediata (R$ 0 / baixa latência):** No modo contínuo mãos-livres, o modo *"Standby"* ignora conversas paralelas até você dizer `"PETeleco"`, abrindo uma janela rolante de **60 s de Escuta Ativa**. Na RTX 3050 CUDA, o SystemOne (`tev1:0.8b`) roteia a fala em **~353 ms** (`p50`); em notebooks só com CPU (`--roteador auto`), o roteador JSON `Gemini Flash-Lite` decide em `~0,7–1,9 s` contra `~10–20 s` no Ollama em CPU.
2. **Resposta social instantânea (`0 ms` de LLM):** Saudações, confirmações de wake-word, despedidas e elogios disparam respostas/áudios curtos da persona do PETeleco sem depender de geração em nuvem.
3. **Funções determinísticas + TTS natural PT-BR:** Data/hora, clima, bateria/temperatura e identificação facial usam handlers determinísticos com síntese neural **`Edge TTS`** (`pt-BR-AntonioNeural`, ritmo natural de conversa) ou **`Gemini TTS`**, com fallback offline em CPU (**`Piper TTS`**).
4. **Travas probabilísticas de segurança motora:** Comandos físicos exigem confiança calibrada (`>= 0,80` execução direta; `0,50–0,79` pede confirmação por voz; parada explícita `>= 0,90` corta imediatamente).
5. **Nuvem estritamente sob demanda:** A API multimodal (`Gemini`) só entra quando o roteador classifica necessidade real de `Image-to-Text` ou conversa aberta com humor orgânico.

---

## 🏗️ Arquitetura em Camadas

```mermaid
flowchart TD
    subgraph L0["Camada 0 — Corpo Físico & I/O (Compartilhado com CaioHVectorA/Rover)"]
        MIC["Entrada de Áudio (Mic PC na V1 / INMP441 I2S planejado V2)"]
        CAM["ESP32-S3-CAM (OV5640) Stream UDP (:5005 'RVR1')"]
        CV["Visão Local: YuNet + SFace (128-D) & YOLOv8n"]
        MOT["Atuadores do Chassi: Serial <C,a,b,c> -> Motores & Pan/Tilt"]
        SPK["Saída de Áudio (PC na V1 / MAX98357A I2S planejado V2)"]
    end

    subgraph L1["Camada 1 — Escuta Local (Planejada, R$ 0)"]
        VAD["Silero VAD (Corte de Silêncio)"]
        STT["faster-whisper (STT PT-BR em CUDA)"]
    end

    subgraph L2["Camada 2 — Roteador Cognitivo Jev-Based (Implementada ✅)"]
        STATE["Snapshot de Estado: Fala + Modo + Face ID + Objetos YOLO"]
        SYS1["Ollama POST /v1/systemone (tev1:0.8b Q8_0 em CUDA)"]
        GATE["Arbitragem & Trava de Segurança (Limiares + Parada)"]
    end

    subgraph L3["Camada 3 — Execução Local, Persona & Memória"]
        POOL["Pool de Áudios .wav da Persona (0 ms TTS)"]
        DET["Handlers Determinísticos + Piper TTS (CPU)"]
        MEM["Memória Biográfica Persistente (SQLite / JSON por Rosto)"]
    end

    subgraph L4["Camada 4 — Cognição Multimodal em Nuvem (Sob Demanda)"]
        GEM["Gemini API (Frame <= 480 px JPEG Assíncrono / Conversa Aberta)"]
    end

    MIC --> VAD --> STT --> STATE
    CAM --> CV --> STATE
    STATE --> SYS1 --> GATE

    GATE -->|"Conversa paralela / TV"| IGN["Ignora Silenciosamente"]
    GATE -->|"pool_fala_pronta"| POOL --> SPK
    GATE -->|"fala_parametrizada"| DET --> SPK
    GATE -->|"acao_rover >= 0.80"| MOT
    GATE -->|"acao_rover 0.50..0.79"| CONF["Pede Confirmação por Voz"] --> SPK
    GATE -->|"cognicao_gemini"| GEM --> DET
    CV <--> MEM
    MEM <--> GEM
```

---

## 🧠 Modelos Utilizados & Orçamento de VRAM

Projetado para manter **todos os modelos ativos residentes dentro de 8 GB de VRAM** (máquina-alvo: **NVIDIA GeForce RTX 3050 OEM 8 GB** + Intel Core i5-14600 + 16 GB DDR5):

| Camada | Componente / Modelo | Função | Execução | Status |
|---|---|---|---|---|
| **C1 (Áudio)** | `VAD RMS (pre-roll 450 ms)` + `Gemini Audio STT` / `faster-whisper` (`small`) + Wake-word `"PETeleco"` | Transcrição PT-BR (`Standby <-> Escuta Ativa 60 s`) + biometria vocal **GE2E Neural 256-D** (`ge2e_256d` na CPU `0 MB` VRAM, confirmação em 2 faixas com SNR + fallback `192-D`) | Cloud STT / CPU / CUDA (`auto`) | **✅ Implementado e Testado** |
| **C2 (Roteador)** | **`tev1:0.8b`** (`772.85M`, `Q8_0`, `/v1/systemone`, **9 perguntas paralelas**) + JSON `Gemini Flash-Lite` (`auto` CPU) | Avaliação paralela `noul`/`choice` (`+cerebro_persona`, `+skill_modular`) e travas de segurança | Ollama CUDA (`1245,7 MiB`) / API | **✅ Implementado e Benchmarked** |
| **C3 (Cérebros & Skills)** | **5 Prompts Multi-Cérebro** (`resenha`, `desgracado`, `tecnico`, `serio`, `utilitario`) + **9 Skills Estilo Alexa** | Few-shot real `user`/`model` + `clima`, `musica` headless sem anúncios (`PyAV + sounddevice` com **audio ducking de 18%**), `noticias`, `recados`, `timer`, `busca_web` grátis, `sonoplastia` (`.wav`), `listas`, `rotinas` | Python local (`0 MB` VRAM) | **✅ Implementado e Testado** |
| **C3 (Voz)** | **`Edge Neural TTS`** (`pt-BR-AntonioNeural`) / `Gemini TTS` + fallback **`Piper TTS`** offline | Síntese conversacional PT-BR via `sounddevice` / Pulse / ALSA | Neural Online / CPU Offline (`0 MB` VRAM) | **✅ Implementado e Testado** |
| **C3 (Visão)** | `YuNet` + `SFace` (ONNX, `128-D`) + `WebcamLive` HUD | Detecção facial, biometria (`>= 0,363`) e preview ao vivo `--preview` / MJPEG | OpenCV ONNX local (`0 MB` VRAM) | **✅ Implementado (`Rover` + `PETeleco`)** |
| **C3 (Visão)** | `YOLOv8n` + `MediaPipe Hands` | Objetos, máquina de estados trekking e gestos | Python / ONNX local | Validado no `Rover` |
| **C3 (Memória)** | **Grafo de Conhecimento Embarcado (`GrafoMemoriaSQLite`)** + Ficha JSON Quente | Subgrafos em 1–2 saltos (`grafo_nos`/`grafo_arestas` em `memoria_longa.sqlite`), extrator regex local (`F1 = 0,941`, `0,18 ms`, `0` tokens) + modo híbrido (`F1 = 0,947`) | Disco local (`data/`, `0 MB` VRAM) | **✅ Implementado e Benchmarked** |
| **C4 (Nuvem)** | `Gemini REST` + fallback Ollama `/api/chat` | `Image-to-Text` da cena e conversa aberta roteada pelo Jev entre os 5 cérebros | HTTPS REST API / local | **✅ Implementado (`GEMINI_API_KEY` opcional)** |

---

## 🔌 Resumo dos Protocolos

- **Contrato da Camada 2 (`POST /v1/systemone`):** Envia snapshot `state` + **9 perguntas tipadas em paralelo** (`dirigido_ao_robo`, `rota_principal`, `pool_fala_pronta`, `fala_parametrizada`, `acao_rover`, `cognicao_gemini`, `precisa_imagem_camera`, `cerebro_persona`, `skill_modular`), devolvendo escolhas e probabilidades calibradas em uma única passada.
- **Streaming UDP Zero-Buffer (`Porta 5005`, herdado do `Rover`):** Cabeçalho binário `'RVR1'` (`0x52565231`, 16 bytes `<IHHHHI`), descoberta via broadcast `HELLO`, ping RTT e dreno contínuo para ler sempre o quadro mais recente sem fila acumulada.
- **Protocolo Serial ASCII (`<C,a,b,c>`, herdado do `Rover`):** Quadros curtos para motores diferenciais (`<M,esq,dir>`), Pan/Tilt suavizado (`<S,pan10,tilt10,vel10>`), bateria (`<B,mV,minmV>`) e parada de emergência (`<X>`).

Consulte [`docs/ARCHITECTURE.pt-BR.md`](docs/ARCHITECTURE.pt-BR.md) ([English](docs/ARCHITECTURE.md)) para especificações completas e regras de arbitragem.

---

## 📊 Resultados Reais de Benchmark — Camada 2 (Verificado)

Medido em **Windows 11 / RTX 3050 OEM (8 GB VRAM) / Ollama v0.40.16 (`OLLAMA_VULKAN=0` forçando CUDA0)** executando [`src/benchmark_router.py`](src/benchmark_router.py) sobre **36 frases em PT-BR** (**20 padrão + 16 inéditas de generalização**, 2 rodadas warm = `72` chamadas):

| Métrica | Resultado Medido |
|---|---|
| **Latência Warm (`n = 72`, 7 perguntas paralelas)** | `min = 306,21 ms` \| `média = 352,85 ms` \| **`p50 = 352,54 ms`** \| **`p95 = 376,05 ms`** \| `max = 491,71 ms` |
| **VRAM Ocupada (`/api/ps`)** | **`1245,7 MiB`** (`keep_alive=-1`) |
| **Acurácia da Rota Principal** | **`36 / 36 (100,0%)`** (`20/20` padrão + `16/16` inéditas) |
| **Acurácia do Subtipo** | **`31 / 31 (100,0%)`** *(excluindo os 5 casos de ruído sem subtipo)* |
| **Detecção de Câmera (`precisa_imagem_camera`)** | **`34 / 34 (100,0%)`** |
| **Resolução Local (Economia de API)** | **`30 / 36 (83,3%)`** resolvidos sem chamar o Gemini |

> **Escopo e comparação Linux vs. Windows:** A tabela acima mede o roteador isolado em **RTX 3050 CUDA** (`352,5 ms` p50). Em execução CPU pura no notebook de desenvolvimento (**Linux / AMD Ryzen 7 5700U / Ollama v0.40.16 CPU**, `n=20`), a acurácia manteve **`20/20 (100,0%)`** com latência warm `p50 = 9,73 s` (`min = 6,34 s` | `média = 9,22 s`) — não ative `OLLAMA_IGPU_ENABLE=1` em APUs AMD Renoir RADV, pois o driver Vulkan trava após `load_tensors`.  
> **Nota prática no Windows:** Por padrão no Windows, o Ollama `v0.40.16` sobe em backend Vulkan (`14,19 s` compilando shaders na 1ª chamada). Defina `OLLAMA_VULKAN=0` antes do `ollama serve` para forçar `CUDA0`. A primeira carga fria pós-boot em CUDA leva ~27–45 s (`mmap` desativado em `windows_cuda`), mantendo-se residente em seguida (`496,1 ms` no warm-reload).

---

## 💻 Projeção de Consumo de Hardware & Cota de Nuvem (Arquitetura Atual)

Consumo medido/projetado para a arquitetura completa (**Multi-Cérebro + 9 Skills Estilo Alexa + Grafo em SQLite + Voz Neural GE2E + Roteador Jev + Visão/STT/TTS**) no PC servidor (**NVIDIA RTX 3050 8 GB VRAM + Intel Core i5-14600 + 16 GB DDR5**):

| Subsistema / Camada | VRAM (`RTX 3050 8 GB`) | RAM do Sistema | CPU / Latência de Inferência | Disco Local |
|---|---|---|---|---|
| **C1 — VAD + `faster-whisper small`** (`float16` CUDA; `0 MiB` via `Gemini Áudio` em CPU) | `~1050–1200 MiB` *(ou `0 MiB`)* | `~350 MB` | `~350–650 ms` warm em CUDA | `~485 MB` |
| **C1 — Biometria Vocal Neural GE2E** (`ge2e_256d`, LSTM 3 camadas em CPU) | **`0 MiB`** | `~45 MB` | `~25–45 ms` / fala | `17,1 MB` (`voice_encoder_ge2e.pt`) |
| **C2 — Roteador Jev `tev1:0.8b`** (`Q8_0`, Ollama `CUDA0`, `OLLAMA_VULKAN=0`, `keep_alive=-1`) | **`1245,7 MiB`** | `~400 MB` | **`p50 = 388,5 ms`** (9 perguntas paralelas) | `~810 MB` (`C:\Ollama`) |
| **C3 — Visão Local (`YuNet + SFace` OpenCV DNN CPU)** | **`0 MiB`** | `~60 MB` | **`~10,8 ms`** / frame | `~37 MB` |
| **C3 — Multi-Cérebro + Grafo SQLite + 9 Skills** (`GrafoMemoriaSQLite`) | **`0 MiB`** | `< 25 MB` | **`0,18 ms`** extração local + `< 1,5 ms` consulta 2-hop | `< 5 MB` (`~5–10 MB/ano`) |
| **C3 — Música Headless (`PyAV + sounddevice`, Ducking 18%) & TTS (`Edge` / `Piper` CPU)** | **`0 MiB`** | `~80 MB` | `< 2%` de 1 núcleo (música) / `~260–400 ms` (Piper) | `~60 MB` (`Piper` + `.wav` SFX) |
| **Consumo Total Residente no PC Servidor** | **`~2,3–2,5 GB` (~30% de 8 GB; ~5,5 GB livres)** | **`~1,1–1,4 GB` (< 9% de 16 GB)** | **Roteamento & skills em sub-segundo** | **`~1,4 GB` totais** |

- **Orçamento de Cota Cloud (`Gemini 3.5 Flash-Lite` Free Tier):**
  - `~78–83%` dos turnos (`ignorar`, `pool_fala_pronta`, `fala_parametrizada`, `acao_rover`, `timer`, `listas`, `recados`, `musica`, `sonoplastia`) gastam **`0` chamadas na nuvem**.
  - Conversa aberta ou narração de utilidades consomem **1 única chamada REST** (`~1050–1320` tokens de entrada + `~20–85` tokens de saída — economia de ~25–35% em tokens de entrada frente ao prompt monolítico graças a Multi-Cérebro + subgrafo em 2 saltos).
  - Extração de grafo em modo `local` (`F1 = 0,941` em bateria de 12 casos) e `busca_web` (`AwesomeAPI` + `ddgs` + `Wikipedia`) consomem **`0` chamadas extras pagas**.
- **Orçamento Energético Embarcado (`Rover` V1 → PCB V2 em `2S 7.4V 2500mAh` + `Mini 560 5V/5A`):**
  - Toda inferência de IA roda no PC servidor. No robô embarcado, `ESP32-S3-CAM` (`~180–260 mA @ 5V`), `ESP32 DevKit V1` da V2 (`~120 mA`), `INMP441` (`~1 mA`), `SSD1306` (`~15 mA`) e o amplificador I2S `MAX98357A` (`~150–300 mA` médios em música ambiente a 18–35% de volume, pico `~600 mA` a 3W) cabem com ampla folga no regulador `5,0V / 5A`.

---

## 📂 Estrutura do Repositório

```text
PETeleco/
├── README.md                   # Documentação em Inglês
├── README.pt-BR.md             # Documentação em Português (BR)
├── AGENTS.md                   # Diretrizes para agentes & contexto paralelo (EN)
├── AGENTS.pt-BR.md             # Diretrizes para agentes & contexto paralelo (PT-BR)
├── .env.example                # Modelo das variáveis de ambiente (GEMINI_API_KEY etc.)
├── requirements.txt            # Dependências de runtime (skills, áudio headless, STT/TTS) + notas PyTorch
├── prompts/
│   ├── persona_peteleco.md     # Prompt monolítico legado de referência (v8.0)
│   └── cerebros/               # Prompts modulares Multi-Cérebro roteados pelo Jev
│       ├── resenha.md          # Parceiro leve e espirituoso (cérebro casual padrão)
│       ├── desgracado.md       # Zoeira ácida sem filtro (bloqueado para professor/SERIO)
│       ├── tecnico.md          # Assistente técnico de engenharia e elétrica (temp=0,35)
│       ├── serio.md            # Trava respeitosa formal (Professor / modo SERIO)
│       └── utilitario.md       # Narração natural estilo Alexa com DADOS_SKILL vivos
├── data/
│   ├── perfis_pessoas.json     # Ficha quente dos perfis e apelidos do PET Elétrica
│   ├── known_faces.json        # Vetores faciais SFace 128-D (YuNet + SFace)
│   └── known_voices.json       # Banco de embeddings vocais GE2E 256-D para identificação por voz
├── docs/
│   ├── context.md              # Memória técnica contínua e restrições físicas (EN)
│   ├── context.pt-BR.md        # Memória técnica contínua e restrições físicas (PT-BR)
│   ├── ARCHITECTURE.md         # Protocolos, travas e modelos (Inglês)
│   └── ARCHITECTURE.pt-BR.md   # Protocolos, travas e modelos (Português-BR)
├── tests/
│   ├── test_arena_prompts.py       # Testes de regressão da Arena de Prompts
│   ├── test_studio_prompt.py       # Testes da API REST e SSE do Prompt Studio
│   └── test_arquitetura_modular.py # Testes Multi-Cérebro, 9 Skills, Grafo SQLite & Voz GE2E (28/28 OK)
└── src/
    ├── __init__.py
    ├── escuta_stt.py               # Camada 1: Microfone (VAD pre-roll + ducking) + STT (Gemini Áudio / faster-whisper)
    ├── voz_biometria.py            # Camada 1: Biometria Vocal Neural 256-D GE2E em CPU + SNR 2 faixas (`192-D` fallback)
    ├── router_systemone.py         # Camada 2: Roteador Jev (/v1/systemone, 9 perguntas) + JSON Flash-Lite + Escuta Ativa
    ├── cerebros.py                 # Camada 3: Carregador Multi-Cérebro, frontmatter, few-shot real & travas de segurança
    ├── grafo_memoria.py            # Camada 3: Grafo de Conhecimento SQLite (`grafo_nos`/`grafo_arestas`, subgrafo 2-hop)
    ├── memoria_pessoas.py          # Camada 3: Memória em 3 camadas (JSON quente + SQLite frio + gancho de Grafo)
    ├── tts_local.py                # Camada 3: Edge Neural / Gemini TTS + Piper TTS offline + ducking de música
    ├── visao_webcam.py             # Camada 3: Captura contínua WebcamLive + janela HUD YuNet/SFace
    ├── persona_peteleco.py         # Camada 3/4: Orquestração Multi-Cérebro, disparo de Skills & Gemini REST
    ├── prompt_metrics.py           # Utilitários de telemetria (tokens, latência e notas)
    ├── testar_peteleco.py          # Bancada & simulador ao vivo CLI
    ├── ui_peteleco.py              # UI Web local (MJPEG YuNet+SFace + SSE "Debaixo dos Panos")
    ├── studio_prompt.py            # Prompt Studio Web (avaliação multi-turno, editor .md Multi-Cérebro & Skills)
    ├── arena_prompts.py            # Servidor automatizado A/B & LLM-as-a-Judge para calibração de prompts
    ├── benchmark_router.py         # Bateria de benchmark isolado do roteador Jev (36 casos)
    ├── benchmark_grafo.py          # Bateria de benchmark Precisão/Recall/F1 do Grafo de Conhecimento (12 casos)
    ├── skills/                     # 9 Skills Modulares Estilo Alexa (`SkillBase`)
    │   ├── __init__.py
    │   ├── base.py
    │   ├── busca_web.py            # Cotações BRL grátis (AwesomeAPI) + ddgs + Wikipedia PT
    │   ├── clima.py                # Clima ao vivo em Niterói/RJ (Open-Meteo, sem chave de API)
    │   ├── listas.py               # Listas de compras e laboratório persistidas no grafo (0 ms LLM)
    │   ├── musica.py               # Player headless sem anúncios (PyAV+sounddevice, ytmusicapi, SoundCloud, SomaFM, ducking 18%)
    │   ├── noticias.py             # Manchetes RSS ao vivo (G1, Tecnoblog, GE)
    │   ├── recados.py              # Recados em SQLite indexados por pessoa (entrega automática ao identificar)
    │   ├── rotinas.py              # Rotinas estilo Alexa ("bom_dia", "modo_estudo")
    │   ├── sonoplastia.py          # Sintetizador determinístico de SFX .wav + trava anti-cringe
    │   └── timer.py                # Temporizadores em thread de fundo com aviso falado
    └── static/
        ├── ui_peteleco.html        # Frontend da UI Web principal
        ├── studio_prompt.html      # Frontend do Prompt Studio
        └── arena_prompts.html      # Frontend da Arena de Prompts
```

---

## 🚀 Como Rodar a Pipeline Completa (Linux CPU ou Windows CUDA)

1. Instale o Ollama (`>= v0.35`, testado na `v0.40.16`) e baixe o modelo `tev1:0.8b`:
   ```bash
   ollama pull tev1:0.8b
   ```
2. Suba o servidor do Ollama:
   - **Linux (CPU / notebook sem GPU NVIDIA):**
     ```bash
     ollama serve
     ```
   - **Windows (PC com RTX 3050 — force CUDA para evitar Vulkan):**
     ```powershell
     $env:OLLAMA_VULKAN="0"
     ollama serve
     ```
3. Crie o ambiente virtual Python e instale a pilha completa:
   - **Linux (CPU leve, sem baixar ~3 GB de wheels CUDA):**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
     pip install -r requirements.txt numpy sounddevice faster-whisper opencv-python piper-tts edge-tts
     ```
     *(Se usar `uv`: `uv pip install --python .venv/bin/python ...`)*
   - **Windows (CUDA / GPU NVIDIA):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt numpy sounddevice faster-whisper opencv-python piper-tts edge-tts
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```
4. *(Recomendado para STT `Gemini`, roteamento rápido em CPU e VLM na nuvem; modo 100% offline também suportado)* Copie `.env.example` para `.env` na raiz (ou use `../rover/.env`) e preencha `GEMINI_API_KEY`. Sem chave, todas as rotas locais (`pool_fala_pronta`, `fala_parametrizada`, `acao_rover` com trava de bancada, webcam `YuNet+SFace`, `faster-whisper small`, `tev1:0.8b` e `Piper TTS`) seguem funcionando offline.
5. Rode a **UI Web completa ao vivo (Webcam MJPEG com YuNet+SFace + Debaixo dos Panos + Escuta Ativa + TTS Neural PT-BR)** ou a bancada CLI:
   ```bash
   # UI Web local (sobe Ollama automaticamente, abre navegador + webcam + seletores ao vivo):
   python -m src.ui_peteleco

   # Ou via bancada CLI mãos-livres (diga "PETeleco" para abrir janela de 60 s de Escuta Ativa):
   python -m src.testar_peteleco --preview --continuo --tts

   # Benchmark isolado da Camada 2 (36 casos):
   python src/benchmark_router.py --suite todas --rodadas 2
   ```
   - **Na primeira execução:** `YuNet`, `SFace` (`models/`) e a voz offline `pt_BR-faber-medium` (`models/piper/`) são baixados automaticamente como backup offline.
   - **Opções de STT, TTS e Roteador:** `--roteador auto` usa Jev em GPUs CUDA e `Gemini Flash-Lite` em notebooks só com CPU (`--roteador jev` força Ollama local). `--stt-modo auto` usa `Gemini Áudio` primeiro na CPU (com fallback em `faster-whisper small`); `--tts-modo auto` usa `Edge Neural` (`pt-BR-AntonioNeural`, alternável na UI ou via `--tts-voz`) com fallbacks `Gemini TTS` e `Piper TTS` offline.
   - **Wake-word e calibragem do microfone:** No modo contínuo (`--continuo` ou botão na UI), falas paralelas são ignoradas em *Standby* até você dizer **`"PETeleco"`**, abrindo uma janela rolante de **60 s** em *Escuta Ativa* (desative com `--sem-wake-word` ou pelo botão `Wake "PETeleco"`). Ajuste a sensibilidade com `--mic <indice>` (`/mics`) e `--limiar-mic 0.012`.
   - **Comandos úteis dentro da bancada:** `[ENTER]` vazio para falar 1 turno no microfone, `/continuo` para alternar escuta contínua, `/preview` para abrir/fechar o HUD da webcam, `/rosto <nome>` e `/voz <nome>` para cadastrar seu rosto e voz, `/corpo <chao|bancada|no_colo>` para testar a trava anti-queda, `/foto <arquivo> [pergunta]` para enviar imagem e `/perfil` para inspecionar a memória em 3 camadas.
