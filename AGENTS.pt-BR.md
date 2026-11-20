# 🤖 Instruções para Agentes de IA (`AGENTS.pt-BR.md` — `PetEletricaUFF/PETeleco`)

[🇺🇸 Read in English](AGENTS.md) | 🇧🇷 **Português (BR)**

Este documento define as diretrizes obrigatórias de engenharia, regras de comportamento e o contexto físico/científico compartilhado para qualquer agente de IA trabalhando no repositório **`PetEletricaUFF/PETeleco`**.

---

## 🔗 1. Arquitetura Paralela em Dois Repositórios: `PETeleco` + `CaioHVectorA/Rover`

O **`PETeleco`** integra uma pesquisa ativa conduzida em **dois repositórios paralelos que compartilham exatamente o mesmo robô físico, bancada e servidor local de IA**:

| Repositório | Escopo Dedicado | Caminho Local / Remoto |
|---|---|---|
| **`PetEletricaUFF/PETeleco`** *(este repo)* | **Cérebro Cognitivo, Persona & Protocolos:** Simulação de senciência / expressão de persona, roteamento probabilístico Jev-based (`/v1/systemone`), pipeline de áudio (VAD/STT/TTS), memória biográfica persistente vinculada a embeddings faciais e travas de segurança para decisão. | [`PETeleco`](https://github.com/PetEletricaUFF/PETeleco) (`c:\Users\caihe\Documents\develop\PETeleco`) |
| **`CaioHVectorA/Rover`** *(repo irmão)* | **Corpo Físico & Visão Embarcada:** Chassi 4WD em ABS impresso em 3D, placas PCB no KiCad, firmwares (`ESP32-S3-CAM` + controlador do chassi), protocolo serial `<C,a,b,c>`, vídeo UDP zero-buffer e ferramentas standalone de visão computacional (`YuNet + SFace`, `YOLOv8n`, `MediaPipe Hands`). | [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover) (`c:\Users\caihe\Documents\develop\rover`) |

### Regra Estrita de Isolamento
* **NUNCA modifique, apague, prepare (`git add`), faça commit ou push dentro de `CaioHVectorA/Rover` (`c:\Users\caihe\Documents\develop\rover`) enquanto estiver trabalhando em `PETeleco`** (e vice-versa), exceto quando o usuário autorizar expressamente o repositório alvo.
* Mantenha `PETeleco` limpo, simples e focado estritamente em **camadas de IA, protocolos, persona, memória e pipeline** — consumindo os contratos físicos do `Rover` sem duplicar arquivos KiCad, desenhos 3D, planilhas de compras ou firmwares C++.

---

## 🚫 2. Regra de Git (Somente com Autorização Explícita)

* **Nunca execute `git add`, `git commit`, `git push` nem crie branches automaticamente** durante sessões normais de brainstorm, ajuste ou depuração.
* Execute operações de gravação no Git **exclusivamente quando o usuário solicitar de forma explícita** (ex.: *"suba no GitHub"*, *"pode commitar"*).

---

## ⚙️ 3. Contexto Físico & Computacional Fixo (Inalterável Sem Validação)

Nunca desconsidere ou reverta estes fatos já validados:

1. **Corpo Físico Compartilhado (`Rover` V1 Montado → PCB V2 Planejada):**
   * **Tração & Vão Livre:** 4x motores DC cilíndricos com redução metálica **JGA16-050** (16 mm de diâmetro, eixo D de 3 mm — **NÃO são N20**), driver ponte-H MOSFET duplo **TB6612FNG**, **rodas de 32 mm** (cubo montado para fora), garantindo **8,0 mm de vão livre real** ($16\text{ mm} - 8\text{ mm} = 8{,}0\text{ mm}$, resolvendo os 3,0 mm críticos das antigas rodas de 22 mm).
   * **Alimentação:** Pack Li-Ion `2S 7.4V 2500mAh` com BMS + regulador Step-Down `Mini 560` (`5.0V` / até `5A`).
   * **Cabeça (Pan/Tilt):** 2x micro servos **SG90** carregando o **ESP32-S3-CAM N16R8** (`OV5640` 5 MP, stream UDP zero-buffer na porta `5005`, magic `'RVR1'`).
   * **Transição do Controlador do Chassi (V1 → V2):**
     * **V1 (Montada hoje na bancada):** `Arduino Nano V3` em placa perfurada comunicando via serial invertida (`ESP32 IO43 -> BC337 -> Nano D0` a `9600 bps`, quadros ASCII `<C,a,b,c>`).
     * **V2 (PCB face única KiCad `rover_pcb_v2_chassi` projetada, aguardando fabricação):** Substitui o Nano por **ESP32 DevKit V1 (30 pinos)** comunicando com a cabeça via **UART de hardware direta em 3.3V** (conector `J2` 4 vias: `GND`, `+5V`, `GPIO16/CAM-TX`, `GPIO23/CAM-RX` — ESP-NOW descartado para evitar disputa no rádio 2.4 GHz durante o stream UDP), somando áudio digital `I2S` (**`INMP441`** mic + **`MAX98357A`** amp 3W), OLED I2C **`SSD1306`**, temperatura **`DS18B20`** e fita **`WS2812B`**.
   * **Restrições Práticas para Desenvolvimento da IA:**
     * **Sem hardware I2S na V1 ainda:** Todos os testes iniciais de voz e escuta (VAD, STT, TTS, pool de áudio da persona) devem rodar usando **microfone e saída de áudio do próprio PC host**.
     * **Sem sensor ToF/ultrassom no chassi ainda:** A aproximação visual de objetos depende apenas da câmera (quando o Pan vira muito para o lado, há ponto cego frontal no chassi).

2. **PC Hospedeiro de Inferência (Servidor Local / Bancada):**
   * **Máquina:** Dell Inc. XPS 8960 — Windows 11 Home Single Language 64-bit
   * **CPU / RAM:** Intel Core i5-14600 (14 núcleos / 20 threads) | 16 GB DDR5-5600
   * **GPU Dedicada:** NVIDIA GeForce **RTX 3050 OEM (8 GB VRAM)**
   * **Orçamento Estrito de VRAM:** Manter **todos os modelos ativos residentes dentro dos 8 GB de VRAM** (meta total `~2,5–3,0 GB` somando STT + Roteador + Visão local) para não cair em swap na RAM e preservar resposta em sub-segundo.
   * **Regra Operacional CUDA no Windows (Ollama):** O Ollama portátil (`v0.40.16` em `C:\Ollama`) inicia por padrão em Vulkan no Windows se não for sobrescrito (`14,19 s` compilando shaders na 1ª chamada). **Sempre iniciar com `OLLAMA_VULKAN=0`** para rodar `tev1:0.8b` em **`CUDA0`** nativo (`1245,7 MiB` VRAM, `keep_alive=-1`). A primeira carga fria pós-boot em CUDA no Windows leva ~27–45 s (porque o Ollama desativa `mmap` em `windows_cuda`), sustentando ~353 ms depois disso.
   * **Desacoplamento & Portabilidade:** Preferir camadas desacopladas por API HTTP/socket leves (evitar `transformers` monolítico no processo principal quando runtimes dedicados como Ollama `/v1/systemone` ou OpenCV ONNX atendem com menor sobrecarga).

---

## 🎓 4. Enquadramento Acadêmico, Integridade Documental & Tom

* **Terminologia Acadêmica:** A proposta criativa envolve um robô com **persona** e **"simulação de senciência"**; em artigos, relatórios e textos técnicos, enquadrar como **Robótica Cognitiva / Embodied AI / Interação Humano-Robô (HRI)** — destacando contribuições mensuráveis (roteamento hierárquico orientado a custo/latência, redução de chamadas em nuvem e travas probabilísticas de segurança antes de acionar motores).
* **Pares Bilíngues Sincronizados:**
  * Visão Geral: [`README.md`](README.md) (EN) ↔ [`README.pt-BR.md`](README.pt-BR.md) (PT-BR)
  * Regras de Agentes: [`AGENTS.md`](AGENTS.md) (EN) ↔ [`AGENTS.pt-BR.md`](AGENTS.pt-BR.md) (PT-BR)
  * Memória Técnica: [`docs/context.md`](docs/context.md) (EN) ↔ [`docs/context.pt-BR.md`](docs/context.pt-BR.md) (PT-BR)
  * Arquitetura & Protocolos: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (EN) ↔ [`docs/ARCHITECTURE.pt-BR.md`](docs/ARCHITECTURE.pt-BR.md) (PT-BR)
* **Rigor & Fidelidade:** Atualizar `docs/context.md` e `docs/context.pt-BR.md` a cada avanço validado. Separar sempre com clareza **o que já foi implementado/medido** (Camada 1 STT + biometria vocal GE2E 256-D em CPU, Camada 2 roteador `/v1/systemone` de 9 perguntas + bateria de 36 casos, Camada 3 Multi-Cérebro com 5 prompts + 9 skills estilo Alexa + Grafo SQLite com `benchmark_grafo.py` + Edge/Piper TTS + Webcam `YuNet+SFace` e UIs Web `ui_peteleco`, `studio_prompt`, `arena_prompts`), **o que já roda no repo irmão `Rover`** (hardware físico, UDP e firmwares) e a **ponte final de despacho UDP/serial para os motores físicos**.
* **Tom de Comunicação:** Direto, prático, um passo de bancada por vez, mantendo código e documentação claros e fiéis aos fatos observados.
