#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camada 3/4 — Gerenciador de Perfis e Memoria Evolutiva em 3 Camadas.

Camada 1 (Ficha Quente em `data/perfis_pessoas.json`, teto fixo ~120 tokens/pessoa):
    - Flags comportamentais (`papel`, `nivel_intimidade`, `tom_com_a_pessoa`,
      `libera_palavrao`, `libera_zoeira_pesada`, `usa_bordao_chefe`).
    - Maximo de 5 bullets em `tracos_personalidade`, 1 resumo curto em
      `resumo_convivencia` (<= 350 chars) e janela FIFO de 5 `ultimos_episodios`.

Camada 2 (Arquivo Frio em `data/memoria_longa.sqlite`):
    - Historico acumulativo completo com busca textual sob demanda (`0 tokens`
      gastos no prompt padrao).

Camada 3 (Consolidacao & Resolucao de Apelidos STT):
    - Corrige apelidos e erros foneticos (`veio`->`jhonatan`, `to2`->`heitor`,
      `guinust`->`knust`, `Felipe Sadi`->`felipe_sass`) e evolui a memoria
      conforme conhece cada pessoa, protegendo travas do professor orientador.
"""

from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
import unicodedata


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PERFIS_PATH = ROOT_DIR / "data" / "perfis_pessoas.json"
DEFAULT_SQLITE_PATH = ROOT_DIR / "data" / "memoria_longa.sqlite"


def carregar_env() -> Dict[str, str]:
    """Carrega variaveis de `.env` local (ou do irmao `../rover/.env`) sem sobrescrever o ambiente."""
    env_vars: Dict[str, str] = {}
    candidatos = [
        ROOT_DIR / ".env",
        ROOT_DIR.parent / "rover" / ".env",
    ]
    for path in candidatos:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for linha in f:
                s = linha.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in env_vars:
                    env_vars[k] = v
    for k, v in os.environ.items():
        if v:
            env_vars[k] = v
    return env_vars

MAX_TRACOS = 5
MAX_EPISODIOS_QUENTES = 5
MAX_RESUMO_CHARS = 350

_STOPWORDS_NOMES = {
    "eu", "tu", "voce", "ele", "ela", "um", "uma", "o", "a", "os", "as",
    "que", "quem", "aqui", "ali", "la", "sim", "nao", "talvez", "cara",
    "mano", "irmao", "brother", "chefe", "robo", "peteleco", "rover",
    "tudo", "bem", "bom", "boa", "oi", "ola", "fala", "eai", "de", "do",
    "da", "dos", "das", "no", "na", "em", "por", "para", "pra", "com",
    "calouro", "veterano", "aluno", "professor", "doido", "maluco", "triste",
    "feliz", "cansado", "ocupado", "pronto", "novo", "nova", "mesmo",
}

_PALAVROES_LEVES_MEDIOS = (
    "porra", "caralho", "puta que pariu", "merda", "cacete", "foda",
    "foder", "fodido", "fdp", "bosta", "cuzao", "sacanagem", "desgraca",
)
_PISTA_ZOEIRA_PESADA = (
    "safado", "safada", "tarado", "tarada", "putaria", "gostosa", "gostoso",
    "pelado", "pelada", "besteira", "sacana", "cachorro", "pika", "pica",
    "rola", "buceta", "xota", "cu", "transar", "pegar geral",
)
_PISTA_TECNICA = (
    "circuito", "circuitos", "eletrica", "placa", "solda", "pcb", "esp32",
    "arduino", "servo", "motor", "osciloscopio", "codigo", "firmware",
    "python", "matlab", "simulink", "proteus", "kicad", "pid", "controle",
    "eletromag", "maquinas", "supercondutor", "maglev", "tensao", "corrente",
    "resistor", "capacitor", "regulador", "multimetro", "bateria", "uart",
)


def normalizar_chave(texto: str) -> str:
    """Remove acentos, pontuacao extra e converte para minusculas."""
    decomp = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomp if not unicodedata.combining(c))
    limpo = re.sub(r"[^a-z0-9\s]", " ", sem_acento.lower())
    return re.sub(r"\s+", " ", limpo).strip()


def _limpar_nome_face(face_id: str) -> str:
    """Extrai nome puro de strings como 'Caio (94%)' ou 'nenhuma'."""
    if not face_id:
        return ""
    limpo = re.sub(r"\([^)]*\)", "", face_id).strip()
    if normalizar_chave(limpo) in ("", "nenhuma", "nenhum", "none", "desconhecido", "unknown"):
        return ""
    return limpo


class GerenciadorMemoriaPessoas:
    """Gerencia ficha quente JSON, arquivo frio SQLite e evolucao de personalidade."""

    def __init__(
        self,
        perfis_path: Optional[Path] = None,
        sqlite_path: Optional[Path] = None,
    ) -> None:
        self.perfis_path = Path(perfis_path or DEFAULT_PERFIS_PATH)
        self.sqlite_path = Path(sqlite_path or DEFAULT_SQLITE_PATH)
        self._lock = threading.Lock()
        self.perfis: Dict[str, Dict[str, Any]] = {}
        self._indice_apelidos: Dict[str, str] = {}
        self._carregar_perfis()
        self._inicializar_sqlite()

    def _carregar_perfis(self) -> None:
        if not self.perfis_path.exists():
            self.perfis_path.parent.mkdir(parents=True, exist_ok=True)
            self.perfis = {
                "desconhecido": self._template_desconhecido(),
            }
            self._salvar_perfis()
        else:
            with self.perfis_path.open("r", encoding="utf-8") as f:
                self.perfis = json.load(f)
        if "desconhecido" not in self.perfis:
            self.perfis["desconhecido"] = self._template_desconhecido()
        self._reconstruir_indice_apelidos()

    def _salvar_perfis(self) -> None:
        self.perfis_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.perfis_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.perfis, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp_path.replace(self.perfis_path)

    def _inicializar_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.sqlite_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pessoa_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    fala_usuario TEXT NOT NULL,
                    resposta_robo TEXT NOT NULL,
                    resumo TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodios_pessoa ON episodios (pessoa_id, id DESC)"
            )
            conn.commit()

    @staticmethod
    def _template_desconhecido() -> Dict[str, Any]:
        return {
            "nome": "desconhecido",
            "apelidos_stt": [],
            "papel": "visitante_nao_cadastrado",
            "nivel_intimidade": 1,
            "tom_com_a_pessoa": "amigavel_observador",
            "libera_palavrao": False,
            "libera_zoeira_pesada": False,
            "usa_bordao_chefe": True,
            "tracos_personalidade": [
                "Ainda nao identificado por rosto ou voz; conversar normalmente e aprender seu estilo aos poucos"
            ],
            "resumo_convivencia": "",
            "ultimos_episodios": [],
        }

    def _reconstruir_indice_apelidos(self) -> None:
        indice: Dict[str, str] = {}
        for pessoa_id, dados in self.perfis.items():
            if pessoa_id == "desconhecido":
                continue
            candidatos = [pessoa_id, dados.get("nome", "")] + list(dados.get("apelidos_stt", []))
            for cand in candidatos:
                chave = normalizar_chave(str(cand))
                if chave:
                    indice[chave] = pessoa_id
        self._indice_apelidos = indice

    def mapear_identificador(self, identificador_bruto: str) -> Optional[str]:
        """Resolve ID, nome oficial ou apelido fonetico para o pessoa_id canonico."""
        chave = normalizar_chave(identificador_bruto)
        if not chave:
            return None
        if chave in self._indice_apelidos:
            return self._indice_apelidos[chave]
        # Tenta match parcial apenas em chaves de duas ou mais palavras quando comecam iguais
        palavras = chave.split()
        if len(palavras) >= 2:
            duas = " ".join(palavras[:2])
            if duas in self._indice_apelidos:
                return self._indice_apelidos[duas]
        if palavras and palavras[0] in self._indice_apelidos:
            return self._indice_apelidos[palavras[0]]
        return None

    def detectar_apresentacao_verbal(self, transcricao: str) -> Optional[str]:
        """Detecta quando o usuario se identifica na fala (ex.: 'aqui e o Manel', 'sou o to2')."""
        texto = (transcricao or "").strip()
        if not texto:
            return None

        padroes = (
            r"(?:aqui\s+[eé]\s+(?:o|a)|sou\s+(?:o|a)|eu\s+sou\s+(?:o|a)|meu\s+nome\s+[eé]|pode\s+me\s+chamar\s+de)\s+([A-Za-zÀ-ÿ0-9]+(?:\s+[A-Za-zÀ-ÿ0-9]+)?)",
            r"(?:[eé]\s+(?:o|a))\s+([A-Za-zÀ-ÿ0-9]+(?:\s+[A-Za-zÀ-ÿ0-9]+)?)\s+(?:aqui|falando)\b",
        )
        for padrao in padroes:
            m = re.search(padrao, texto, flags=re.IGNORECASE)
            if not m:
                continue
            candidato_bruto = m.group(1).strip()
            # Tenta 2 palavras (ex: 'Felipe Sass', 'jon jon') e depois 1 palavra (ex: 'to2', 'Manel')
            partes = candidato_bruto.split()
            for tentativa in (candidato_bruto, partes[0]):
                pid = self.mapear_identificador(tentativa)
                if pid:
                    return pid
            # Se nao existe no roster ainda e parece um nome proprio plausivel, cria perfil novo
            primeira_norm = normalizar_chave(partes[0])
            if (
                primeira_norm
                and len(primeira_norm) >= 3
                and primeira_norm not in _STOPWORDS_NOMES
                and not primeira_norm.isdigit()
            ):
                return self.garantir_pessoa(partes[0].capitalize())
        return None

    def garantir_pessoa(self, nome: str, papel: str = "visitante_cadastrado") -> str:
        """Cria um perfil blank caso ainda nao exista para o nome informado."""
        pid_existente = self.mapear_identificador(nome)
        if pid_existente:
            return pid_existente

        pid = normalizar_chave(nome).replace(" ", "_") or "visitante"
        with self._lock:
            if pid not in self.perfis:
                self.perfis[pid] = {
                    "nome": nome.strip(),
                    "apelidos_stt": [nome.strip()],
                    "papel": papel,
                    "nivel_intimidade": 2,
                    "tom_com_a_pessoa": "em_observacao",
                    "libera_palavrao": False,
                    "libera_zoeira_pesada": False,
                    "usa_bordao_chefe": True,
                    "tracos_personalidade": [],
                    "resumo_convivencia": "",
                    "ultimos_episodios": [],
                }
                self._salvar_perfis()
                self._reconstruir_indice_apelidos()
        return pid

    def obter_perfil(self, pessoa_id: str) -> Dict[str, Any]:
        return dict(self.perfis.get(pessoa_id) or self.perfis["desconhecido"])

    def listar_aliases_stt(self) -> List[str]:
        """Lista todos os nomes e apelidos para melhorar o initial_prompt do STT."""
        vistos: List[str] = []
        seen = set()
        for pid, dados in self.perfis.items():
            if pid == "desconhecido":
                continue
            for item in [dados.get("nome", "")] + list(dados.get("apelidos_stt", [])):
                s = str(item).strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    vistos.append(s)
        return vistos

    def resolver_pessoa(
        self,
        transcricao: str = "",
        face_id: str = "nenhuma",
        voice_id: str = "desconhecido",
        voice_conf: float = 0.0,
    ) -> Tuple[str, Dict[str, Any], str, str]:
        """
        Funde visao (`SFace`), voz (`IdentificadorVoz`) e apresentacao verbal.

        Retorna `(pessoa_id, perfil_dict, modo_identificacao, pessoa_display)`.
        `modo_identificacao` assume: `rosto+voz`, `apenas_rosto`, `apenas_voz`,
        `apresentacao_verbal` ou `desconhecido`.
        """
        face_nome = _limpar_nome_face(face_id)
        face_pid = self.mapear_identificador(face_nome) if face_nome else None
        voice_pid = self.mapear_identificador(voice_id) if voice_id and voice_id != "desconhecido" else None
        verbal_pid = self.detectar_apresentacao_verbal(transcricao)

        if face_pid and voice_pid:
            if face_pid == voice_pid:
                pid = face_pid
                modo = "rosto+voz"
            elif face_pid == "felipe_sass":
                # Trava de seguranca: se a camera enxerga o orientador, respeita imediatamente
                pid = face_pid
                modo = "apenas_rosto"
            else:
                pid = face_pid
                modo = "apenas_rosto"
        elif face_pid:
            pid = face_pid
            modo = "apenas_rosto"
        elif voice_pid:
            pid = voice_pid
            modo = "apenas_voz"
        elif verbal_pid:
            pid = verbal_pid
            modo = "apresentacao_verbal"
        else:
            pid = "desconhecido"
            modo = "desconhecido"

        perfil = self.obter_perfil(pid)
        nome_oficial = perfil.get("nome", pid)
        if pid == "desconhecido":
            display = "desconhecido"
        elif modo == "apenas_voz" and voice_conf > 0:
            display = f"{nome_oficial} ({int(round(voice_conf * 100))}% voz)"
        elif modo in ("rosto+voz", "apenas_rosto") and "(" in face_id:
            display = f"{nome_oficial} ({face_id.split('(', 1)[1].rstrip(')')})"
        else:
            display = nome_oficial

        return pid, perfil, modo, display

    def formatar_contexto_prompt(self, pessoa_id: str) -> Tuple[str, str]:
        """Formata a Ficha Quente com teto fixo de tamanho para o prompt do LLM."""
        p = self.obter_perfil(pessoa_id)
        tracos = p.get("tracos_personalidade") or []
        tracos_str = "; ".join(str(t) for t in tracos[:MAX_TRACOS]) if tracos else "ainda em branco (observando)"
        perfil_str = (
            f"nome={p.get('nome', pessoa_id)}, papel={p.get('papel', 'membro_pet')}, "
            f"intimidade={p.get('nivel_intimidade', 3)}/5, tom={p.get('tom_com_a_pessoa', 'em_observacao')}, "
            f"libera_palavrao={str(bool(p.get('libera_palavrao', False))).lower()}, "
            f"libera_zoeira_pesada={str(bool(p.get('libera_zoeira_pesada', False))).lower()}, "
            f"usa_bordao_chefe={str(bool(p.get('usa_bordao_chefe', True))).lower()}, "
            f"tracos=[{tracos_str}]"
        )

        resumo = (p.get("resumo_convivencia") or "").strip()
        episodios = p.get("ultimos_episodios") or []
        partes_mem: List[str] = []
        if resumo:
            partes_mem.append(f"Resumo: {resumo[:MAX_RESUMO_CHARS]}")
        if episodios:
            partes_mem.append("Recentes: " + " | ".join(str(e) for e in episodios[-MAX_EPISODIOS_QUENTES:]))
        memoria_str = " ; ".join(partes_mem) if partes_mem else "vazio"
        return perfil_str, memoria_str

    def registrar_interacao(
        self,
        pessoa_id: str,
        fala_usuario: str,
        resposta_robo: str,
        resumo_episodio: Optional[str] = None,
        consolidador_llm: Optional[Callable[[Dict[str, Any], str, str], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Salva episodio no arquivo frio SQLite e atualiza a Ficha Quente JSON.

        `consolidador_llm` (opcional) pode retornar um dict com
        `tracos_personalidade`, `resumo_convivencia`, `libera_zoeira_pesada`, etc.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M")
        data_curta = agora[:10]
        fala_limpa = (fala_usuario or "").strip()
        resp_limpa = (resposta_robo or "").strip()
        if not fala_limpa:
            return

        resumo_curto = (resumo_episodio or self._resumir_episodio_local(fala_limpa, resp_limpa)).strip()

        with closing(sqlite3.connect(self.sqlite_path)) as conn:
            conn.execute(
                """
                INSERT INTO episodios (pessoa_id, timestamp, fala_usuario, resposta_robo, resumo)
                VALUES (?, ?, ?, ?, ?)
                """,
                (pessoa_id, agora, fala_limpa, resp_limpa, resumo_curto),
            )
            conn.commit()

        if pessoa_id == "desconhecido" or pessoa_id not in self.perfis:
            return

        with self._lock:
            perfil = self.perfis[pessoa_id]
            episodios = list(perfil.get("ultimos_episodios") or [])
            episodios.append(f"{data_curta}: {resumo_curto}")
            perfil["ultimos_episodios"] = episodios[-MAX_EPISODIOS_QUENTES:]

            # Protecao estrita: nunca relaxa regras do professor orientador
            if perfil.get("papel") != "professor_orientador":
                self._evoluir_tracos_heuristico(perfil, fala_limpa)
                if consolidador_llm is not None:
                    sugestao = consolidador_llm(dict(perfil), fala_limpa, resp_limpa)
                    if isinstance(sugestao, dict):
                        self._aplicar_consolidacao_segura(perfil, sugestao)

            self._salvar_perfis()

    def buscar_memoria_fria(self, pessoa_id: str, termo: str = "", limite: int = 5) -> List[Dict[str, str]]:
        """Consulta episodios antigos do SQLite sob demanda sem poluir o prompt padrao."""
        with closing(sqlite3.connect(self.sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            if termo.strip():
                like = f"%{termo.strip()}%"
                rows = conn.execute(
                    """
                    SELECT timestamp, fala_usuario, resposta_robo, resumo
                    FROM episodios
                    WHERE pessoa_id = ? AND (fala_usuario LIKE ? OR resumo LIKE ? OR resposta_robo LIKE ?)
                    ORDER BY id DESC LIMIT ?
                    """,
                    (pessoa_id, like, like, like, limite),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT timestamp, fala_usuario, resposta_robo, resumo
                    FROM episodios
                    WHERE pessoa_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (pessoa_id, limite),
                ).fetchall()
        return [dict(r) for r in rows]

    def obter_historico_dialogo(self, pessoa_id: str, limite_turnos: int = 4) -> List[Dict[str, str]]:
        """Retorna os ultimos turnos de uma pessoa do SQLite em ordem cronologica para manter linearidade."""
        if not self.sqlite_path.exists():
            return []
        with closing(sqlite3.connect(self.sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT fala_usuario, resposta_robo
                FROM episodios
                WHERE pessoa_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (pessoa_id, max(1, limite_turnos)),
            ).fetchall()
        historico: List[Dict[str, str]] = []
        for r in reversed(rows):
            fala = str(r["fala_usuario"] or "").strip()
            resp = str(r["resposta_robo"] or "").strip()
            if fala and resp:
                historico.append({"role": "user", "text": fala})
                historico.append({"role": "model", "text": resp})
        return historico

    def obter_contexto_sala_recente(self, limite: int = 3) -> str:
        """Retorna as ultimas falas recentes na sala (mesmo entre pessoas diferentes) para continuidade."""
        if not self.sqlite_path.exists():
            return "vazio"
        with closing(sqlite3.connect(self.sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT timestamp, pessoa_id, fala_usuario, resposta_robo
                FROM episodios
                ORDER BY id DESC LIMIT ?
                """,
                (max(1, limite),),
            ).fetchall()
        if not rows:
            return "vazio"
        linhas: List[str] = []
        for r in reversed(rows):
            pid = str(r["pessoa_id"] or "desconhecido")
            nome = self.obter_perfil(pid).get("nome", pid)
            hora = str(r["timestamp"] or "")[-5:] or "--:--"
            fala = re.sub(r"\s+", " ", str(r["fala_usuario"] or "")).strip()
            resp = re.sub(r"\s+", " ", str(r["resposta_robo"] or "")).strip()
            if len(fala) > 90:
                fala = fala[:87].rstrip() + "..."
            if len(resp) > 120:
                resp = resp[:117].rstrip() + "..."
            if fala and resp:
                linhas.append(f'[{hora}] {nome}: "{fala}" -> PETeleco: "{resp}"')
        return " | ".join(linhas) if linhas else "vazio"

    @staticmethod
    def _resumir_episodio_local(fala: str, resposta: str = "") -> str:
        """Compacta pergunta do usuario e resposta do PETeleco em uma linha para a janela FIFO."""
        fala_limpa = re.sub(r"\s+", " ", fala).strip()
        if len(fala_limpa) > 65:
            fala_limpa = fala_limpa[:62].rstrip() + "..."
        resp_limpa = re.sub(r"\s+", " ", resposta or "").strip()
        if not resp_limpa:
            return fala_limpa
        if len(resp_limpa) > 75:
            resp_limpa = resp_limpa[:72].rstrip() + "..."
        return f"Disse: '{fala_limpa}' -> PETeleco: '{resp_limpa}'"

    @staticmethod
    def _evoluir_tracos_heuristico(perfil: Dict[str, Any], fala_usuario: str) -> None:
        """Atualiza tracos e flags de estilo conforme observa o jeito da pessoa."""
        fala_norm = normalizar_chave(fala_usuario)
        tracos = list(perfil.get("tracos_personalidade") or [])
        preservar_tracos_criador = (
            perfil.get("papel") == "membro_pet_criador" and len(tracos) >= MAX_TRACOS
        )

        def _adicionar_traco(novo: str) -> None:
            if preservar_tracos_criador:
                return
            if novo not in tracos:
                tracos.append(novo)

        usou_palavrao = any(p in fala_norm for p in _PALAVROES_LEVES_MEDIOS)
        usou_zoeira_pesada = any(p in fala_norm for p in _PISTA_ZOEIRA_PESADA)
        assunto_tecnico = any(p in fala_norm for p in _PISTA_TECNICA)

        if usou_palavrao:
            perfil["libera_palavrao"] = True
            _adicionar_traco("fala solto com palavrao na conversa")
        if usou_zoeira_pesada:
            perfil["libera_palavrao"] = True
            perfil["libera_zoeira_pesada"] = True
            perfil["tom_com_a_pessoa"] = "zoeira_sem_freio"
            _adicionar_traco("zoeiro/boca-suja, gosta de provocar e aceita devolucao pesada")
        if assunto_tecnico:
            _adicionar_traco("conversa sobre bancada, projetos ou materias de eletrica")

        # Sobe intimidade gradualmente com interacoes acumuladas (teto 5)
        intimidade_atual = int(perfil.get("nivel_intimidade", 3))
        if usou_palavrao or usou_zoeira_pesada:
            intimidade_atual = min(5, max(intimidade_atual, 4))
        perfil["nivel_intimidade"] = intimidade_atual

        if perfil.get("tom_com_a_pessoa") == "em_observacao" and len(tracos) >= 2:
            perfil["tom_com_a_pessoa"] = "parceiro_pet"

        perfil["tracos_personalidade"] = tracos[-MAX_TRACOS:]
        if not perfil.get("resumo_convivencia") and tracos:
            perfil["resumo_convivencia"] = (
                f"{perfil.get('nome', 'Colega')} do PET; " + ", ".join(tracos[:2])
            )[:MAX_RESUMO_CHARS]

    @staticmethod
    def _aplicar_consolidacao_segura(perfil: Dict[str, Any], sugestao: Dict[str, Any]) -> None:
        """Aplica atualizacao sintetizada mantendo os tetos da Ficha Quente."""
        if perfil.get("papel") == "professor_orientador":
            return
        if isinstance(sugestao.get("tracos_personalidade"), list) and sugestao["tracos_personalidade"]:
            perfil["tracos_personalidade"] = [
                str(t).strip() for t in sugestao["tracos_personalidade"][:MAX_TRACOS] if str(t).strip()
            ]
        if isinstance(sugestao.get("resumo_convivencia"), str) and sugestao["resumo_convivencia"].strip():
            perfil["resumo_convivencia"] = sugestao["resumo_convivencia"].strip()[:MAX_RESUMO_CHARS]
        for flag in ("libera_palavrao", "libera_zoeira_pesada"):
            if isinstance(sugestao.get(flag), bool):
                perfil[flag] = sugestao[flag]
        if isinstance(sugestao.get("tom_com_a_pessoa"), str) and sugestao["tom_com_a_pessoa"].strip():
            perfil["tom_com_a_pessoa"] = sugestao["tom_com_a_pessoa"].strip()
