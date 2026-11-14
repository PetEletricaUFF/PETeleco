# 🤖 Instruções para Agentes de IA (`AGENTS.md` — `PetEletricaUFF/PETeleco`)

Este documento define as diretrizes obrigatórias e o contexto compartilhado para qualquer agente de IA trabalhando no repositório **`PETeleco`**.

---

## 🔗 1. Desenvolvimento Paralelo: `PETeleco` + `CaioHVectorA/Rover`

Este projeto faz parte de uma pesquisa em andamento conduzida em **dois repositórios paralelos que compartilham o mesmo contexto físico e hardware**:

| Repositório | Escopo Dedicado | Caminho Local / URL |
|---|---|---|
| **`PetEletricaUFF/PETeleco`** *(este repo)* | **Cérebro Cognitivo & IA:** Persona ("robô senciente" / *Embodied AI* / HRI), roteamento probabilístico hierárquico (Jev-based), pipeline de fala (VAD/STT/TTS), memória biográfica persistente por pessoa e protocolos de decisão/segurança. | [`PETeleco`](https://github.com/PetEletricaUFF/PETeleco) (`c:\Users\caihe\Documents\develop\PETeleco`) |
| **`CaioHVectorA/Rover`** *(repo irmão)* | **Corpo Físico & Visão Embarcada:** Mecânica 4WD (chassi 3D ABS), PCB V2 no KiCad, firmwares (`ESP32-S3-CAM` + controlador do chassi), enlace serial `<C,a,b,c>`, streaming UDP zero-buffer e módulos práticos de visão computacional. | [`CaioHVectorA/Rover`](https://github.com/CaioHVectorA/rover) (`c:\Users\caihe\Documents\develop\rover`) |

### Regra Crítica de Isolamento
* **NUNCA modifique, apague arquivos ou execute comandos Git (`git add`, `commit`, `push`, `checkout`) dentro de `c:\Users\caihe\Documents\develop\rover` enquanto estiver trabalhando em `PETeleco`** (e vice-versa), salvo se o usuário autorizar expressamente o repositório específico.
* Trate `PETeleco` como repositório limpo e simples focado exclusivamente em **IA, protocolos, persona e pipeline**, consumindo os contratos físicos definidos em `CaioHVectorA/Rover` sem duplicar arquivos CAD, KiCad, compras ou firmwares C++.

---

## 🚫 2. Regra de Git (Apenas com Autorização Explícita)

* **Não faça `git commit` ou `git push` automaticamente** em trocas normais de desenvolvimento/brainstorm.
* Só execute comandos de gravação no Git (`git add`, `git commit`, `git push`) quando o usuário pedir explicitamente.

---

## ⚙️ 3. Contexto Físico & Hardware Compartilhado (Inalterável Sem Validação)

1. **Corpo Físico Atual (`Rover` V1 Montado → Transição PCB V2):**
   * Chassi 4WD em ABS (rodas 32 mm, vão livre 8,0 mm), 4x motores DC `JGA16-050`, driver `TB6612FNG`, cabeça Pan/Tilt (2x `SG90`) com `ESP32-S3-CAM` (`OV5640`, vídeo UDP porta `5005`).
   * **V1 (atual na bancada):** `Arduino Nano V3` em placa perfurada comandando motores/servos via protocolo ASCII `<C,a,b,c>`.
   * **V2 planejada (`rover_pcb_v2_chassi`):** Substitui o Nano por `ESP32 DevKit V1 (30p)` e integra áudio `I2S` (`INMP441` microfone + `MAX98357A` amplificador), OLED `SSD1306` e temperatura `DS18B20`.
   * **Restrição física de teste de voz:** Como o hardware `I2S` ainda não está fisicamente montado na V1, todos os testes iniciais de escuta e fala (VAD / STT / TTS) usam **microfone e saída de áudio do próprio PC host**.
   * **Restrição sensorial de navegação:** Sem sensor ToF/ultrassom no chassi até o momento; a aproximação de objetos depende da câmera na cabeça móvel.

2. **Máquina Hospedeira de Inferência (PC Host):**
   * **PC:** Dell Inc. XPS 8960 — Windows 11 Home Single Language 64-bit
   * **CPU / RAM:** Intel Core i5-14600 (14 núcleos / 20 threads) | 16 GB DDR5-5600
   * **GPU:** NVIDIA GeForce **RTX 3050 OEM (8 GB VRAM)**
   * **Orçamento estrito de VRAM:** Manter modelos residentes confortavelmente dentro dos **8 GB de VRAM** (~2,5–3,0 GB somados planejados) para evitar swap em RAM e manter latência sub-segundo.
   * **Descoberta operacional (Ollama no Windows):** Sempre iniciar o daemon com **`OLLAMA_VULKAN=0`** para forçar o backend nativo **`CUDA0`** (evitando `14,19 s` de compilação de shaders Vulkan na 1ª chamada).

---

## 📖 4. Enquadramento Acadêmico, Documentação & Fidelidade

* **Enquadramento:** A proposta criativa de "robô senciente com persona" deve ser enquadrada academicamente como **Robótica Cognitiva / Embodied AI / Interação Humano-Robô (HRI)** — evitando afirmar "senciência literal" em textos científicos e destacando os diferenciais reais: arquitetura hierárquica orientada a custo/latência e travas probabilísticas de segurança antes de acionar motores.
* **Documentação Bilíngue:** Manter [`README.md`](README.md) (EN) e [`README.pt-BR.md`](README.pt-BR.md) (PT-BR), bem como [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) e [`docs/ARCHITECTURE.pt-BR.md`](docs/ARCHITECTURE.pt-BR.md), sincronizados e fiéis ao estado real do código.
* **Memória Técnica Contínua:** Registrar marcos, benchmarks e decisões em [`docs/context.md`](docs/context.md). Nunca relatar testes ou componentes planejados como se já estivessem validados em hardware.
