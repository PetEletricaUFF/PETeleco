# 🏗️ Especificação de Arquitetura & Protocolos (`PETeleco`)

[🇺🇸 Read in English](ARCHITECTURE.md) | 🇧🇷 **Português (BR)**

Este documento detalha o pipeline cognitivo, os contratos entre camadas, as travas probabilísticas de segurança, a camada de persona/memória e os protocolos físicos de comunicação do **`PetEletricaUFF/PETeleco`**.

---

## 1. Arquitetura Dual: Corpo (`Rover`) vs. Cérebro (`PETeleco`)

O **`PETeleco`** e o **[`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover)** evoluem em paralelo compartilhando o mesmo robô físico e o mesmo PC servidor:

- **`CaioHVectorA/Rover` (Plataforma Física):**
  - Chassi 4WD em ABS impresso em 3D (rodas 32 mm, vão livre 8,0 mm, 4x motores `JGA16-050`, driver `TB6612FNG`).
  - Cabeça Pan/Tilt (2x servos `SG90` + `ESP32-S3-CAM OV5640`) com vídeo UDP de baixa latência.
  - Controlador do chassi em transição da **V1** (`Arduino Nano V3` em placa perfurada) para a **PCB V2** (`rover_pcb_v2_chassi` face única em KiCad com `ESP32 DevKit V1 30p`, microfone I2S `INMP441`, amplificador I2S `MAX98357A`, OLED `SSD1306` e sensor `DS18B20`).
  - *Restrições físicas atuais para testes:* Como o áudio `I2S` ainda não está montado na V1, os primeiros testes de escuta e fala utilizam o **microfone e alto-falantes do PC host**. Além disso, sem sensor ToF/ultrassom no chassi até o momento, a estimativa de distância depende apenas da câmera.
- **`PetEletricaUFF/PETeleco` (Núcleo Cognitivo):**
  - Consome os contratos físicos do `Rover` e concentra exclusivamente **persona, roteamento hierárquico local-first, pipeline de voz, memória biográfica por rosto e escalonamento controlado para nuvem**.

---

## 2. Pipeline Cognitivo & de Persona Ponta a Ponta

```text
Áudio Mic ──▶ Silero VAD ──▶ faster-whisper (CUDA PT-BR) ──┐
                                                            ▼
Vídeo UDP ──▶ YuNet+SFace / YOLOv8n / Estado Rover ──▶ ContextoRover.formatar_state()
                                                            │
                                                            ▼
                                               POST /v1/systemone (tev1:0.8b)
                                                            │
        ┌───────────────────┬───────────────────────┬───────┴───────────┬─────────────────────┐
        ▼                   ▼                       ▼                   ▼                     ▼
   [ignorar]       [pool_fala_pronta]      [fala_parametrizada]   [acao_rover]       [cognicao_gemini]
   (Ruído/TV)       (.wav pré-gravado)      (Handler local +       (Direto >=0,80 /   (Gemini Assíncrono
   Descarta          0 ms de síntese         Piper TTS + Cache)     Conf. 0,50..0,79)  + Foto se preciso)
```

### Camada de Persona & Memória Persistente (Projeto)
- **Imediatismo da Persona (`0 ms` + Síntese Dinâmica):** Interações sociais curtas (`saudacao`, `despedida`, `ok_confirmacao`, `elogio_agradecimento`) sorteiam clipes `.wav` pré-gravados com a voz/jeito do PETeleco, eliminando latência de TTS. Respostas dinâmicas usam **`Piper TTS`** offline na CPU.
- **Memória Biográfica por Rosto (Planejada):** Quando **`YuNet` + `SFace`** (`128-D`, limiar cosseno `>= 0,363`) reconhece a pessoa diante do robô, `LOCAL_FACE_ID` alimenta o contexto e permite consultar/gravar fatos e resumos de conversas passadas em banco **`SQLite` / `JSON`** local indexado pela pessoa.

---

## 3. Protocolo da Camada 2: Contrato `POST /v1/systemone`

Em vez de gerar tokens de texto livre ou JSON instável com LLM convencional, o **PETeleco** usa o endpoint Jev-based **`/v1/systemone`** do Ollama (`>= v0.35`, validado na `v0.40.16`) com **`tev1:0.8b`** (`keep_alive=-1`). Em uma única passada paralela, o modelo avalia `state` contra **7 perguntas tipadas**.

### 3.1 Formato Estruturado de `state` (`ContextoRover.formatar_state()`)
```text
USER_SPEECH (pt-BR): "<transcricao STT>"
CURRENT_ROVER_MODE: <descricao mapeada do modo, ex: idle_ready (motors off, waiting)>
LOCAL_FACE_ID (SFace): <ex: Caio (94%) | desconhecido | nenhuma>
LOCAL_OBJECTS_SEEN (YOLO): <ex: person, bottle>
```

### 3.2 Esquema das 7 Perguntas Paralelas (`construir_perguntas_rover()`)

| ID da Pergunta | Tipo | Classes / Função |
|---|---|---|
| `dirigido_ao_robo` | `noul` | Probabilidade escalar `[0..1]` de que a fala é dirigida ao robô |
| `rota_principal` | `choice` | `ignorar`, `pool_fala_pronta`, `fala_parametrizada`, `acao_rover`, `cognicao_gemini` |
| `pool_fala_pronta` | `choice` | `saudacao`, `despedida`, `ok_confirmacao`, `elogio_agradecimento`, `nenhuma` |
| `fala_parametrizada` | `choice` | `hora_atual`, `previsao_clima`, `bateria_temp`, `quem_vejo_agora`, `nenhuma` |
| `acao_rover` | `choice` | `seguir_rosto`, `ir_ate_objeto`, `parar`, `radar`, `nenhuma` |
| `cognicao_gemini` | `choice` | `descrever_cena`, `conversa_aberta`, `nenhuma` |
| `precisa_imagem_camera` | `noul` | Probabilidade escalar `[0..1]` de necessidade de foto atual da câmera |

---

## 4. Motor de Segurança Probabilística & Arbitragem (`RoteadorSystemOne.decidir()`)

Regras determinísticas em Python auditam as probabilidades do `/v1/systemone` antes de qualquer ação física:

1. **Prioridade Absoluta de Parada de Emergência:**
   - Se `acao_choice == "parar"` com `acao_prob >= 0,90` e `p_dirigido >= 0,50` -> executa `acao_rover:parar` imediatamente (`requer_confirmacao=False`).
2. **Filtro Combinado de Conversas Paralelas & TV:**
   - Descarta silenciosamente (`ignorar:conversa_paralela_ou_ruido`) quando (`limiar_dirigido = 0,55`):
     - `p_dirigido < 0,35`, ou
     - `p_dirigido < 0,55 e p_ignorar >= 0,35`, ou
     - `p_ignorar >= 0,65 e p_dirigido < 0,75`, ou
     - `p_ignorar >= 0,55 e p_dirigido < 0,65`.
3. **Travas de Segurança para Motores (`acao_rover`):**
   - Com `acao_choice != "nenhuma"`:
     - `acao_prob >= 0,80` (`limiar_acao_direta`) -> comando enviado direto (`requer_confirmacao=False`).
     - `acao_prob < 0,80` (exceto `parar`) -> marca `requer_confirmacao=True` para pedir confirmação verbal antes de energizar motores.
   - Se `acao_choice == "nenhuma"`, mas `rota_prob >= 0,90` e `p_dirigido >= 0,65`, verifica se há subação clara (`>= 0,25`); havendo, escala com `requer_confirmacao=True` em vez de perder comando imperativo válido; caso contrário, cai para rotas não motoras para evitar acionamento falso.
4. **Desempate por Especialidade Local:**
   - Se `param_choice != "nenhuma"` com `param_prob >= 0,85`, `rota_choice in ("pool_fala_pronta", "cognicao_gemini")` e `p_imagem < 0,55`, garante roteamento para `fala_parametrizada` local.
5. **Gatilho Visual da Nuvem (`cognicao_gemini`):**
   - Ativa `precisa_imagem_camera = True` quando `p_imagem >= 0,55` ou `subtipo == "descrever_cena"`.

---

## 5. Protocolos Físicos Herdáveis do `CaioHVectorA/Rover`

| Protocolo | Meio / Porta | Formato / Detalhes Técnicos |
|---|---|---|
| **Vídeo UDP Zero-Delay** | UDP Porta `5005` | Magic `0x52565231` (`'RVR1'`), cabeçalho de 16 bytes `<IHHHHI`, descoberta broadcast `HELLO`, ping RTT (`P` + `<d`), dreno de socket sem fila (sempre frame mais recente), comandos `1..9, 0, K, Z`. |
| **Controle Serial Chassi/Cabeça** | UART ASCII `<C,a,b,c>` | `<P,n>`/`<A,n>` ping-ack, `<S,pan10,tilt10[,vel10]>` Pan/Tilt em décimos de grau com velocidade adaptativa, `<M,esq,dir>` motores `-255..255`, `<X>` parada de emergência, `<B,mV,minmV>` bateria, `<H,up_s,rx,err>` heartbeat 1 Hz, `<C,id,valor>` sintonia. |
| **Cliente Multimodal Nuvem** | HTTPS REST | Thread assíncrona em background (`Gemini` API) redimensionando o frame para `<= 480 px` JPEG (qualidade 72) sem travar vídeo nem motores. |

---

## 6. Ambiente de Execução & Reprodutibilidade do Benchmark

- **PC Hospedeiro:** Dell XPS 8960 | Intel Core i5-14600 | 16 GB DDR5-5600 | **NVIDIA GeForce RTX 3050 OEM (8 GB VRAM)** | Windows 11.
- **Vulkan vs. CUDA no Windows:**
  - Por padrão no Windows, o Ollama `v0.40.16` usa backend Vulkan (`OLLAMA_VULKAN=true`), gastando **`14.193,6 ms`** compilando shaders na 1ª chamada (`p50 = 297,9 ms`, `p95 = 321,18 ms` nos testes iniciais de 6 perguntas).
  - Definindo **`OLLAMA_VULKAN=0`**, o servidor utiliza **`CUDA0`** nativo (`Flash Attention`), ocupando **`1245,7 MiB` de VRAM** (`/api/ps`). A carga inicial pós-boot em `windows_cuda` leva ~27–45 s (porque o Ollama desativa `mmap` no Windows CUDA), permanecendo residente em seguida com `keep_alive=-1` (`min = 306,21 ms`, `média = 352,85 ms`, **`p50 = 352,54 ms`**, **`p95 = 376,05 ms`**, `max = 491,71 ms`, `n = 72`).
- **Acurácia Verificada em 36 Casos (`src/benchmark_router.py --suite todas --rodadas 2`):**
  - **20 casos padrão + 16 casos inéditos de generalização** -> `36/36 (100,0%)` em rota principal, `31/31 (100,0%)` em subtipo, `34/34 (100,0%)` na detecção de imagem e **`30/36 (83,3%)`** resolvidos localmente sem chamar a API do Gemini.

---

## 7. Próximos Passos Planejados

1. **Camada 1 (Escuta):** Integrar microfone contínuo local (`Silero VAD` -> `faster-whisper` CUDA) preenchendo `ContextoRover`.
2. **Camada 3 (Persona, Funções & Memória):** Gravar pool `.wav` da persona PETeleco, implementar handlers determinísticos (`hora_atual`, `previsao_clima`, `bateria_temp`, `quem_vejo_agora`), voz offline `Piper TTS` e banco `SQLite` por rosto reconhecido.
3. **Camada 4 + Ponte Física:** Ligar escalonamento `Gemini` e despachar ações confirmadas ao `CaioHVectorA/Rover` via UDP/serial.
