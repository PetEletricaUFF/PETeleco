# 📖 Memória Técnica & Contexto de Pesquisa — `PETeleco`

> *English summary available in [`README.md`](../README.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).*  
> Registro contínuo das decisões de arquitetura, modelos, protocolos e benchmarks do projeto **`PetEletricaUFF/PETeleco`**, desenvolvido em paralelo com [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover).

---

## 1. Origem, Propósito & Relação com `CaioHVectorA/Rover`

- **Inspiração direta:** O `PETeleco` nasce diretamente da plataforma robótica [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover) e divide com ela o **mesmo ambiente físico de bancada, hardware móvel e computador hospedeiro**.
- **Divisão de responsabilidades:**
  - **`CaioHVectorA/Rover`** concentra engenharia mecânica (chassi 3D em ABS), eletrônica/PCB KiCad, firmwares (`ESP32-S3-CAM` + `Arduino Nano V3` → `ESP32 DevKit`), streaming UDP de vídeo (`zero_delay_udp.py`), biometria facial (`YuNet + SFace`), detecção/navegação (`YOLOv8n`) e gestos (`MediaPipe Hands`).
  - **`PetEletricaUFF/PETeleco`** concentra o **núcleo cognitivo**: simulação de senciência e construção de **persona**, roteamento probabilístico local de baixa latência (Jev-based), pipeline conversacional de áudio (VAD → STT → TTS), memória biográfica persistente associada a rostos e especificação dos protocolos entre camadas.

---

## 2. Contexto Físico e Computacional Compartilhado

### Corpo do Robô (Referência `Rover`)
- **V1 em bancada:** Tração 4WD (`4x JGA16-050`, rodas 32 mm com 8,0 mm de vão livre), driver `TB6612FNG`, cabeça Pan/Tilt (`2x SG90`) com `ESP32-S3-CAM` (`OV5640`) e `Arduino Nano V3` em placa perfurada.
- **Evolução planejada (PCB V2 — `rover_pcb_v2_chassi`):** Migra o chassi para `ESP32 DevKit V1 (30 pinos)` com áudio digital `I2S` (`INMP441` microfone + `MAX98357A` amplificador), display `OLED SSD1306` e temperatura `DS18B20`.
- **Limitações físicas atuais relevantes para a IA:**
  1. O hardware de áudio `I2S` ainda **não está montado na V1**; portanto, os testes iniciais de voz e escuta são conduzidos diretamente com o **microfone e saída de áudio do PC host**.
  2. Não há sensor de distância dedicado (ToF/ultrassom) no chassi; o desvio/aproximação visual depende exclusivamente da câmera Pan/Tilt.

### Servidor Local de Inferência (PC Host)
- **Computador:** Dell Inc. XPS 8960 — Windows 11 Home Single Language 64-bit
- **CPU / Memória:** Intel Core i5-14600 (14 cores / 20 threads) | 16 GB DDR5-5600
- **GPU Dedicada:** NVIDIA GeForce **RTX 3050 OEM (8 GB VRAM)**
- **Meta de projeto:** Manter todos os modelos neurais residentes dentro dos **8 GB de VRAM** (`~2,5–3,0 GB` somados) para evitar swap para RAM e eliminar engasgos de interação.

---

## 3. Arquitetura em Camadas & Estado Atual

| Camada | Função | Stack / Modelos | Status no `PETeleco` |
|---|---|---|---|
| **C0 — Corpo & Transporte** | Vídeo baixa latência, servos, tração e telemetria | UDP porta `5005` (`'RVR1'`), Serial `<C,a,b,c>`, `I2S`/`OLED` (V2); áudio via PC na V1 | Operacional no `Rover` (vídeo/visão); áudio físico pendente V2 |
| **C1 — Escuta Local (R$ 0)** | Corte de silêncio e transcrição contínua PT-BR | `Silero VAD` → `faster-whisper` (`base` / `small` em CUDA) | **Planejada** (próxima etapa) |
| **C2 — Roteador Cognitivo (R$ 0)** | Filtro de conversa paralela, seleção de camada, detecção de câmera e trava de segurança motora | Ollama `>= v0.35` (`POST /v1/systemone`) + **`tev1:0.8b`** (`Q8_0`, CUDA) | **Implementada e validada** (`src/router_systemone.py`) |
| **C3 — Execução Local, Persona & Memória (R$ 0)** | Falas prontas `.wav` (persona), handlers determinísticos (`hora`, `clima`, `bateria`, `quem vejo`), `Piper TTS` (CPU), `YuNet + SFace`, `YOLOv8n` e `SQLite`/`JSON` | Visão validada no `Rover`; handlers locais, `Piper TTS` e memória biográfica em integração no `PETeleco` | **Visão no `Rover` / Persona & Memória planejadas** |
| **C4 — Cognição Pesada (Sob Demanda)** | Descrição de cena (`Image-to-Text`) e conversa aberta complexa | **Gemini API** (`gemini-3.5-flash` assíncrono no `Rover`, frame `<= 480 px`) | Cliente assíncrono validado no `Rover`; escalonamento integrado via C2 |

---

## 4. Histórico Técnico & Benchmarks Verificados

### 14/11/2026 — Implantação da Camada 2 (Roteador SystemOne `tev1:0.8b`) e Criação do Repositório `PETeleco`

- **Configuração de Inferência Local (RTX 3050 OEM 8 GB VRAM):**
  - Instalado Ollama portátil **v0.40.16** em `C:\Ollama` com o modelo **`tev1:0.8b`** (`772.85M` parâmetros, arquitetura `qwen35`, quantização `Q8_0`, ~`784,5 MiB` em disco).
  - **Comparativo Vulkan vs CUDA no Windows:** Sem configuração explícita, o `ollama serve` no Windows iniciava com backend Vulkan, consumindo `14.193,6 ms` na primeira chamada para compilação de shaders (`p50 = 297,9 ms`, `p95 = 321,18 ms` com 6 perguntas no benchmark preliminar). Ao definir **`OLLAMA_VULKAN=0`**, o servidor passou a operar nativamente em **`CUDA0`** (`Flash Attention`, `keep_alive=-1`). Em CUDA no Windows, a carga fria pós-boot leva ~27–45 s porque o Ollama desativa `mmap` (`reason=windows_cuda`), mas mantém o modelo residente em **`1245,7 MiB` de VRAM** (`/api/ps`).
- **Benchmark Isolado da Camada 2 (36 frases PT-BR: 20 padrão + 16 inéditas de generalização, 2 rodadas warm = 72 inferências):**
  - **Escopo estrito do teste:** Avaliação de texto + contexto simulado de estado/SFace/YOLO direto via HTTP `POST /v1/systemone` (sem microfone/STT, TTS ou chassi físico conectado).
  - **Cold-start (com modelo já residente):** `496,1 ms`.
  - **Latência warm (`n = 72`, 7 perguntas paralelas):** `min = 306,21 ms` | `média = 352,85 ms` | **`p50 = 352,54 ms`** | **`p95 = 376,05 ms`** | `max = 491,71 ms`.
  - **Acurácia observada:**
    - Rota principal: **`36/36 (100,0%)`** (`20/20` padrão + `16/16` generalização inédita).
    - Subtipo específico: **`31/31 (100,0%)`** *(excluindo os 5 casos `ignorar` sem subtipo esperado)*.
    - Detecção de necessidade de imagem (`precisa_imagem_camera`): **`34/34 (100,0%)`**.
    - Economia de chamadas em nuvem: **`30/36 (83,3%)`** resolvidos localmente sem acionar a API do Gemini.
    - Trava de segurança motora verificada nos testes: comandos intermediários (`acao_seguir = 0,55` e `acao_ir_objeto = 0,50`) acionaram `[CONFIRMAR]` por ficarem abaixo do limiar `0,80`, enquanto `parar` explícito (`0,98`), `radar` (`0,84`) e `gen_acao_1..4` (`0,87`–`1,00`) passaram diretamente.
- **Separação oficial do repositório `PetEletricaUFF/PETeleco`:** Criada a base documental bilíngue (EN / PT-BR) e migrado o roteador + suíte de benchmark para [`src/router_systemone.py`](../src/router_systemone.py) e [`src/benchmark_router.py`](../src/benchmark_router.py).
