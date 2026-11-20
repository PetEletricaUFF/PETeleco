#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pacote de Skills Modulares do PETeleco."""

from pathlib import Path
from typing import Optional

from src.grafo_memoria import GrafoMemoriaSQLite

from .base import ContextoSkill, GerenciadorSkills, ResultadoSkill, SkillBase
from .busca_web import SkillBuscaWeb
from .clima import SkillClima
from .listas import SkillListas
from .musica import SkillMusica
from .noticias import SkillNoticias
from .recados import SkillRecados
from .rotinas import SkillRotinas
from .sonoplastia import SkillSonoplastia
from .timer import SkillTimer


def criar_gerenciador_skills(
    *,
    sqlite_path: Optional[Path] = None,
    sfx_dir: Optional[Path] = None,
    grafo: Optional[GrafoMemoriaSQLite] = None,
    dry_run_os: bool = False,
) -> GerenciadorSkills:
    """Instancia o registro padrão com todas as 9 skills modulares do PETeleco."""
    grafo_efetivo = grafo or GrafoMemoriaSQLite(sqlite_path=sqlite_path)
    sk_rotinas = SkillRotinas(grafo=grafo_efetivo, sqlite_path=sqlite_path)
    gerenciador = GerenciadorSkills(
        skills=[
            SkillClima(),
            SkillMusica(dry_run=dry_run_os),
            SkillNoticias(),
            SkillRecados(sqlite_path=sqlite_path),
            SkillTimer(),
            SkillBuscaWeb(),
            SkillSonoplastia(sfx_dir=sfx_dir, dry_run=dry_run_os),
            SkillListas(grafo=grafo_efetivo, sqlite_path=sqlite_path),
            sk_rotinas,
        ]
    )
    sk_rotinas.vincular_gerenciador(gerenciador)
    return gerenciador


__all__ = [
    "ContextoSkill",
    "GerenciadorSkills",
    "ResultadoSkill",
    "SkillBase",
    "SkillBuscaWeb",
    "SkillClima",
    "SkillListas",
    "SkillMusica",
    "SkillNoticias",
    "SkillRecados",
    "SkillRotinas",
    "SkillSonoplastia",
    "SkillTimer",
    "criar_gerenciador_skills",
]
