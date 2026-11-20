#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arquitetura Multi-Cérebro do PETeleco (Prompts Modulares Roteados por Jev).

Cada "cérebro" (versão/especialidade do PETeleco) vive como um arquivo `.md`
independente em `prompts/cerebros/` com frontmatter simples (`id`, `nome`,
`temperature`, `max_tokens`, `criterio_jev`, etc.) e prompt enxuto sem regras
conflitantes.

A seleção do cérebro ativo combina:
1. Travas determinísticas de segurança (ex.: `professor_orientador` SEMPRE
   trava no cérebro `serio`, nunca caindo em `desgracado`).
2. Contexto de ferramentas (`DADOS_SKILL` -> `utilitario`, visão -> `tecnico`).
3. Roteamento probabilístico Jev-based (`tev1:0.8b` via `/v1/systemone`).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CEREBROS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "cerebros"

REGRAS_BASE_IMUTAVEIS = """=== REGRAS IMUTÁVEIS DE VERDADE & SEGURANÇA FÍSICA ===
- HONESTIDADE SOBRE PESSOAS: Só conhece quem está no contexto/memória (Prof. Felipe Sass, Caio, Heitor/to2, Manuel, Jhonatan, Iury, Luigi, Lívia, Nathan, Knust). Para qualquer outro nome não presente no contexto, afirme claramente que não conhece sem inventar fama ou histórico.
- CORPO INVISÍVEL POR PADRÃO: Nunca cite rodas, câmera, bateria ou laboratório sem pedido físico/visual ou pergunta direta sobre seu hardware (se perguntarem direto: V1 atual tem ESP32-S3-CAM OV5640, Nano V3, TB6612FNG, 4x JGA16-050, 2S 7,4V e Mini 560, sem I2S ativo nem sensor de borda/ToF; V2 planejada traz ESP32 DevKit V1, I2S, OLED 21/22, DS18B20 e D34).
- TRAVAS FÍSICAS: Em apenas_voz, se pedirem pra olhar algo, diga que só está ouvindo e peça pra mostrar na câmera. Em superficie bancada ou no_colo, recuse andar ou ir até borda por risco de queda sem sensor de borda.
- FORMATO DE FALA: Zero markdown, asterisco, crase, emoji ou "chefe" (mesmo se usa_bordao_chefe=true)."""


@dataclass
class CerebroSpec:
    """Definição imutável de um cérebro/versão modular do PETeleco."""

    id: str
    nome: str
    descricao: str
    prompt_base: str
    temperature: float = 0.9
    max_tokens: int = 240
    permite_sonoplastia: bool = False
    bloqueado_para_professor: bool = False
    criterio_jev: str = ""
    path: Optional[Path] = None
    exemplos_few_shot: List[Tuple[str, str]] = field(default_factory=list)

    def para_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nome": self.nome,
            "descricao": self.descricao,
            "prompt_base": self.prompt_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "permite_sonoplastia": self.permite_sonoplastia,
            "bloqueado_para_professor": self.bloqueado_para_professor,
            "criterio_jev": self.criterio_jev,
            "exemplos_few_shot": [{"user": u, "model": a} for u, a in self.exemplos_few_shot],
            "path": str(self.path) if self.path else "",
        }


def extrair_few_shot_dialogo(prompt_texto: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Separa o bloco opcional `=== FEW_SHOT_DIALOGO ===` (`U: ...` / `A: ...`) do
    texto do system prompt para injetá-lo como turnos reais `user`/`model` no
    `contents` do Gemini (muito mais eficaz em modelos Flash-Lite do que bullets).
    """
    marcador = "=== FEW_SHOT_DIALOGO ==="
    if marcador not in (prompt_texto or ""):
        return (prompt_texto or "").strip(), []

    antes, bloco = prompt_texto.split(marcador, 1)
    pares: List[Tuple[str, str]] = []
    u_atual: Optional[str] = None
    for linha in bloco.splitlines():
        ln = linha.strip()
        if not ln:
            continue
        if ln.startswith("==="):
            break
        if ln.startswith("U:"):
            u_atual = ln[2:].strip()
        elif ln.startswith("A:") and u_atual:
            a_atual = ln[2:].strip()
            if a_atual:
                pares.append((u_atual, a_atual))
            u_atual = None
    return antes.strip(), pares


def _parse_bool(valor: str, default: bool = False) -> bool:
    v = (valor or "").strip().lower()
    if v in ("true", "1", "yes", "sim", "on"):
        return True
    if v in ("false", "0", "no", "nao", "não", "off"):
        return False
    return default


def _parse_frontmatter_md(conteudo: str, fallback_id: str, path: Optional[Path] = None) -> CerebroSpec:
    """Faz parse de frontmatter YAML simples (`--- ... ---`) sem depender de PyYAML."""
    meta: Dict[str, str] = {}
    corpo = (conteudo or "").strip()

    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", corpo, flags=re.DOTALL)
    if m:
        bloco_meta = m.group(1)
        corpo = m.group(2).strip()
        for linha in bloco_meta.splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or ":" not in linha:
                continue
            chave, val = linha.split(":", 1)
            meta[chave.strip().lower()] = val.strip().strip('"').strip("'")

    cid = (meta.get("id") or fallback_id).strip().lower()
    nome = meta.get("nome") or f"Cérebro {cid.capitalize()}"
    descricao = meta.get("descricao") or ""
    try:
        temperature = min(2.0, max(0.0, float(meta.get("temperature", "0.9"))))
    except ValueError:
        temperature = 0.9
    try:
        max_tokens = min(2048, max(64, int(meta.get("max_tokens", "240"))))
    except ValueError:
        max_tokens = 240

    _, pares_fs = extrair_few_shot_dialogo(corpo)
    return CerebroSpec(
        id=cid,
        nome=nome,
        descricao=descricao,
        prompt_base=corpo,
        temperature=temperature,
        max_tokens=max_tokens,
        permite_sonoplastia=_parse_bool(meta.get("permite_sonoplastia", "false")),
        bloqueado_para_professor=_parse_bool(meta.get("bloqueado_para_professor", "false")),
        criterio_jev=meta.get("criterio_jev", ""),
        path=path,
        exemplos_few_shot=pares_fs,
    )


def _serializar_cerebro_md(spec: CerebroSpec) -> str:
    """Serializa um CerebroSpec de volta para `.md` preservando frontmatter limpo."""
    linhas = [
        "---",
        f"id: {spec.id}",
        f"nome: {spec.nome}",
        f"descricao: {spec.descricao}",
        f"temperature: {spec.temperature}",
        f"max_tokens: {spec.max_tokens}",
        f"permite_sonoplastia: {'true' if spec.permite_sonoplastia else 'false'}",
        f"bloqueado_para_professor: {'true' if spec.bloqueado_para_professor else 'false'}",
        f"criterio_jev: {spec.criterio_jev}",
        "---",
        spec.prompt_base.strip(),
        "",
    ]
    return "\n".join(linhas)


class GerenciadorCerebros:
    """Carrega, arbitra e compõe prompts modulares dos cérebros do PETeleco."""

    ORDEM_PADRAO = ("resenha", "desgracado", "tecnico", "serio", "utilitario")

    def __init__(
        self,
        diretorio: Optional[Path] = None,
        limiar_troca_jev: float = 0.45,
        limiar_auto_desgracado: float = 0.72,
    ) -> None:
        self.diretorio = Path(diretorio) if diretorio else DEFAULT_CEREBROS_DIR
        self.limiar_troca_jev = float(limiar_troca_jev)
        self.limiar_auto_desgracado = float(limiar_auto_desgracado)
        self._cerebros: Dict[str, CerebroSpec] = {}
        self.recarregar()

    def recarregar(self) -> Dict[str, CerebroSpec]:
        """Lê todos os arquivos `*.md` da pasta de cérebros."""
        carregados: Dict[str, CerebroSpec] = {}
        if self.diretorio.exists():
            for arq in sorted(self.diretorio.glob("*.md")):
                if arq.name.startswith("_") or arq.name.lower() == "readme.md":
                    continue
                texto = arq.read_text(encoding="utf-8")
                spec = _parse_frontmatter_md(texto, fallback_id=arq.stem, path=arq)
                carregados[spec.id] = spec

        if not carregados:
            carregados["resenha"] = CerebroSpec(
                id="resenha",
                nome="Cérebro Resenha",
                descricao="Fallback padrão descontraído.",
                prompt_base="Você é o PETeleco, conversando de forma descontraída e natural em português brasileiro.",
                temperature=1.0,
                max_tokens=200,
                permite_sonoplastia=True,
                bloqueado_para_professor=True,
                criterio_jev="Casual chat, joking around, or open social conversation.",
            )
        self._cerebros = carregados
        return dict(self._cerebros)

    def listar(self) -> List[CerebroSpec]:
        """Retorna os cérebros em ordem determinística (padrões primeiro, extras em ordem alfabética)."""
        vistos = set()
        saida: List[CerebroSpec] = []
        for cid in self.ORDEM_PADRAO:
            if cid in self._cerebros:
                saida.append(self._cerebros[cid])
                vistos.add(cid)
        for cid in sorted(self._cerebros):
            if cid not in vistos:
                saida.append(self._cerebros[cid])
        return saida

    def obter(self, cerebro_id: str, fallback: str = "resenha") -> CerebroSpec:
        cid = (cerebro_id or "").strip().lower()
        if cid in self._cerebros:
            return self._cerebros[cid]
        if fallback in self._cerebros:
            return self._cerebros[fallback]
        return next(iter(self._cerebros.values()))

    def salvar_prompt_cerebro(
        self,
        cerebro_id: str,
        novo_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CerebroSpec:
        """Atualiza o prompt (e hiperparâmetros opcionais) de um cérebro no disco."""
        cid = (cerebro_id or "").strip().lower()
        if cid not in self._cerebros:
            raise KeyError(f"Cérebro '{cid}' não encontrado.")
        corpo = (novo_prompt or "").strip()
        if not corpo:
            raise ValueError("O prompt do cérebro não pode ficar vazio.")

        _, pares_fs = extrair_few_shot_dialogo(corpo)
        atual = self._cerebros[cid]
        atualizado = CerebroSpec(
            id=atual.id,
            nome=atual.nome,
            descricao=atual.descricao,
            prompt_base=corpo,
            temperature=atual.temperature if temperature is None else min(2.0, max(0.0, float(temperature))),
            max_tokens=atual.max_tokens if max_tokens is None else min(2048, max(64, int(max_tokens))),
            permite_sonoplastia=atual.permite_sonoplastia,
            bloqueado_para_professor=atual.bloqueado_para_professor,
            criterio_jev=atual.criterio_jev,
            path=atual.path or (self.diretorio / f"{cid}.md"),
            exemplos_few_shot=pares_fs,
        )
        self.diretorio.mkdir(parents=True, exist_ok=True)
        destino = atualizado.path or (self.diretorio / f"{cid}.md")
        destino.write_text(_serializar_cerebro_md(atualizado), encoding="utf-8")
        self._cerebros[cid] = atualizado
        return atualizado

    def construir_criterios_jev(self) -> Dict[str, str]:
        """Gera dinamicamente o mapa `id -> criterio_jev` para o roteador `/v1/systemone`."""
        criterios: Dict[str, str] = {}
        for spec in self.listar():
            if spec.criterio_jev:
                criterios[spec.id] = spec.criterio_jev
        return criterios

    def resolver_cerebro(
        self,
        *,
        modo_persona: str = "NORMAL",
        perfil: Optional[Dict[str, Any]] = None,
        cerebro_jev: str = "",
        confianca_jev: float = 0.0,
        cerebro_forcado: str = "",
        tem_dados_skill: bool = False,
        precisa_imagem: bool = False,
    ) -> Tuple[CerebroSpec, str]:
        """
        Arbitra qual cérebro atende o turno combinando travas de segurança e Jev.

        Retorna `(CerebroSpec, motivo_decisao)`.
        """
        perfil_seg = perfil or {}
        eh_prof = perfil_seg.get("papel") == "professor_orientador"
        modo_norm = (modo_persona or "NORMAL").strip().upper()

        # 1) Trava formal absoluta: professor orientador ou modo SERIO nunca saem do cérebro 'serio'
        if eh_prof:
            return self.obter("serio"), "trava_professor_orientador"
        if modo_norm == "SERIO":
            return self.obter("serio"), "modo_persona_serio"

        # 2) Pin manual explícito (respeitando bloqueios de segurança)
        forcado = (cerebro_forcado or "").strip().lower()
        if forcado and forcado in self._cerebros:
            spec = self._cerebros[forcado]
            if not (eh_prof and spec.bloqueado_para_professor):
                return spec, f"pin_manual:{forcado}"

        # 3) Quando uma skill modular retornou dados vivos, o cérebro 'utilitario' narra com fidelidade
        if tem_dados_skill and "utilitario" in self._cerebros:
            return self.obter("utilitario"), "skill_dados_ativos"

        # 4) Inspeção visual com foto da câmera -> cérebro 'tecnico' para fidelidade visual
        if precisa_imagem and "tecnico" in self._cerebros:
            return self.obter("tecnico"), "inspecao_visual_camera"

        # 5) Modo DESGRACADO explícito pelo usuário
        if modo_norm == "DESGRACADO" and "desgracado" in self._cerebros:
            return self.obter("desgracado"), "modo_persona_desgracado"

        # 6) Roteamento probabilístico Jev-based no modo NORMAL (auto)
        sug = (cerebro_jev or "").strip().lower()
        if sug and sug in self._cerebros and confianca_jev >= self.limiar_troca_jev:
            if sug == "desgracado":
                # Auto-ativação do cérebro desgraçado em NORMAL exige alta confiança + perfil liberado
                if confianca_jev >= self.limiar_auto_desgracado and bool(perfil_seg.get("libera_zoeira_pesada", False)):
                    return self.obter("desgracado"), f"jev_auto:{sug}({confianca_jev:.2f})"
                return self.obter("resenha"), f"jev_guardrail_desgracado_para_resenha({confianca_jev:.2f})"
            return self.obter(sug), f"jev_auto:{sug}({confianca_jev:.2f})"

        return self.obter("resenha"), "fallback_resenha"

    @staticmethod
    def compor_system_instruction(
        *,
        cerebro: CerebroSpec,
        bloco_contexto: str,
        dados_skill: str = "",
        recados_pendentes: str = "",
        incluir_regras_imutaveis: bool = True,
    ) -> str:
        """Monta o System Prompt enxuto do cérebro ativo (sem o bloco FEW_SHOT, que vai como turnos reais)."""
        prompt_limpo, _ = extrair_few_shot_dialogo(cerebro.prompt_base)
        partes = [prompt_limpo or cerebro.prompt_base.strip()]
        if incluir_regras_imutaveis:
            partes.append(REGRAS_BASE_IMUTAVEIS)
        partes.append(f"CEREBRO_ATIVO: {cerebro.id} ({cerebro.nome})\n" + bloco_contexto.strip())
        if dados_skill.strip():
            partes.append(f"=== DADOS_SKILL (VERDADE EM TEMPO REAL) ===\n{dados_skill.strip()}")
        if recados_pendentes.strip():
            partes.append(
                "=== RECADOS_PENDENTES (ENTREGUE NATURALMENTE NESTE TURNO) ===\n"
                + recados_pendentes.strip()
            )
        return "\n\n".join(partes)
