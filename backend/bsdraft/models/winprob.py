"""Antisymmetric win-probability model: P(team_a beats team_b | map, mode).

Design (a clean inductive bias for drafting):

    logit = [ S(A, ctx) - S(B, ctx) ]          # context-conditioned team strength
          + [ PA·QB - PB·QA ]                   # low-rank antisymmetric counter term
          + [ T(A) - T(B) ]                     # class-level within-team synergy (optional)

where ctx = [map_emb, mode_emb]; S is a shared MLP over the mean brawler embedding + ctx;
and P/Q are low-rank "attacker/defender" embeddings whose dot products encode directed
matchups. Swapping A and B negates the logit, so P(A wins) + P(B wins) = 1 by
construction — no team-order bias and no global offset to learn.

Within-team synergy (``class_synergy``): the strength term above pools a team by the *mean*
of its brawler embeddings, which is order-invariant but also collapses composition — two
tanks average to roughly the tank centroid, so a marginal added brawler contributes almost
independently of its teammates (exactly additive if the strength MLP were linear). That
leaves the net with no channel to learn that some pairs overlap (two immobile throwers, both
dived by one assassin) or combo (a thrower behind a wall-breaker). T supplies that channel at
the *archetype* level: with a learnable symmetric class×class matrix ``M`` and each brawler's
class ``cls(i)``,

    T(team) = sum_{i<j in team} M[cls(i), cls(j)]

pools every same-class pairing in the data into one estimate (far fewer params, far more
games per param than a per-brawler embedding — stable and interpretable). Nothing presumes a
sign: the data decides which archetype pairs help vs overlap. Entering as T(A) - T(B) keeps
the logit antisymmetric (empty board predicts exactly 0.5; mask-vs-mask cancels). NB: the
learned redundancy signal is *weak* (comparable to seed noise for most archetypes — only the
thrower/Artillery anti-synergy reproduces cleanly); see docs/model-evaluation.md.

Partial drafts: when ``mask_row`` is set, every brawler-indexed matrix carries one extra
learned row — the "unknown slot". Training masks slots down to realistic draft states, so
at inference an unfinished board is scored directly: unknown slots marginalize over how
real drafts continued. Antisymmetry is unaffected (mask-vs-mask counter terms cancel
exactly), so an empty board predicts exactly 0.5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from bsdraft.constants import BRAWLER_CLASSES

N_CLASSES = len(BRAWLER_CLASSES)          # 7 archetypes
UNKNOWN_CLASS = N_CLASSES                  # sentinel for the mask row / OOV — excluded from synergy


def brawler_class_rows(num_brawlers: int) -> List[int]:
    """Class index (0..N_CLASSES-1) for each brawler embedding row, in encoder-row order, plus a
    trailing UNKNOWN_CLASS for the mask row. Built from the reference catalog + the fixed
    ``BRAWLER_CLASSES`` order so training and serving agree. Any brawler whose class isn't one of
    the seven (shouldn't happen) also maps to UNKNOWN_CLASS and is excluded from the synergy sum."""
    from bsdraft.data import encoders as E
    from bsdraft.data import reference as R
    cls_idx = {c: i for i, c in enumerate(BRAWLER_CLASSES)}
    cls_by_id = {b.id: b.cls for b in R.load_brawlers()}
    id_by_row = {row: bid for bid, row in E.brawler_encoder().items()}
    rows = [cls_idx.get(cls_by_id.get(id_by_row.get(i)), UNKNOWN_CLASS) for i in range(num_brawlers)]
    rows.append(UNKNOWN_CLASS)  # mask row
    return rows


@dataclass
class ModelConfig:
    num_brawlers: int
    num_maps: int
    num_modes: int
    d_brawler: int = 32
    d_map: int = 16
    d_mode: int = 8
    d_hidden: int = 64
    counter_rank: int = 16
    # Class-level within-team synergy: a learnable symmetric class x class matrix, so every
    # same-archetype pairing in the data feeds ONE estimate (two throwers, two tanks, ...) —
    # pooled, stable, interpretable. Absent from older config dicts, so it defaults off and
    # legacy checkpoints load unchanged (see T(A)-T(B) term in the module docstring).
    class_synergy: bool = False
    dropout: float = 0.1
    # Row index of the trained "unknown slot" embedding (= num_brawlers), or None for a
    # legacy full-comp-only model. Serialized into the export so serving can detect support.
    mask_row: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class WinProbNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # One extra row past the real brawlers holds the "unknown slot" embedding. Its only
        # coherent position is num_brawlers — serving's OOV fallback treats rows [0:mask_row]
        # as "all real brawlers" — so enforce that rather than half-support other values.
        if cfg.mask_row is not None and cfg.mask_row != cfg.num_brawlers:
            raise ValueError(
                f"mask_row must equal num_brawlers ({cfg.num_brawlers}), got {cfg.mask_row}")
        rows = cfg.num_brawlers + (1 if cfg.mask_row is not None else 0)
        self.brawler = nn.Embedding(rows, cfg.d_brawler)
        self.map_emb = nn.Embedding(cfg.num_maps, cfg.d_map)
        self.mode_emb = nn.Embedding(cfg.num_modes, cfg.d_mode)
        self.strength = nn.Sequential(
            nn.Linear(cfg.d_brawler + cfg.d_map + cfg.d_mode, cfg.d_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, 1),
        )
        # low-rank counter embeddings (attacker P, defender Q)
        self.counter_p = nn.Embedding(rows, cfg.counter_rank)
        self.counter_q = nn.Embedding(rows, cfg.counter_rank)
        # class-level synergy: a learnable symmetric class x class matrix + a fixed row->class map.
        self.use_class_synergy = bool(cfg.class_synergy)
        if self.use_class_synergy:
            self.register_buffer("brawler_class",
                                 torch.tensor(brawler_class_rows(cfg.num_brawlers), dtype=torch.long))
            self.class_syn = nn.Parameter(torch.zeros(N_CLASSES, N_CLASSES))  # starts at no effect
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.brawler, self.map_emb, self.mode_emb):
            nn.init.normal_(emb.weight, std=0.1)
        nn.init.normal_(self.counter_p.weight, std=0.05)
        nn.init.normal_(self.counter_q.weight, std=0.05)

    def _ctx(self, map_idx: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.map_emb(map_idx), self.mode_emb(mode_idx)], dim=-1)

    def _strength(self, team: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        team_vec = self.brawler(team).mean(dim=1)  # (B, d_brawler) — order-invariant
        return self.strength(torch.cat([team_vec, ctx], dim=-1)).squeeze(-1)  # (B,)

    def _class_synergy(self, team: torch.Tensor) -> torch.Tensor:
        """Class-level within-team synergy: sum over the (3 choose 2) slot pairs of the symmetric
        class-pair weight, excluding any slot that is the mask row / unknown class. (B,)."""
        M = self.class_syn + self.class_syn.t()                 # symmetrize (order-invariant)
        c = self.brawler_class[team]                            # (B, 3) class idx, mask -> UNKNOWN
        real = c < N_CLASSES                                    # exclude mask / unknown slots
        cc = c.clamp(max=N_CLASSES - 1)                         # safe gather index
        total = team.new_zeros(team.shape[0], dtype=M.dtype)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            both = (real[:, i] & real[:, j]).to(M.dtype)
            total = total + both * M[cc[:, i], cc[:, j]]
        return total

    def forward(
        self,
        team_a: torch.Tensor,
        team_b: torch.Tensor,
        map_idx: torch.Tensor,
        mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self._ctx(map_idx, mode_idx)
        strength = self._strength(team_a, ctx) - self._strength(team_b, ctx)
        pa = self.counter_p(team_a).sum(dim=1)  # (B, r)
        qa = self.counter_q(team_a).sum(dim=1)
        pb = self.counter_p(team_b).sum(dim=1)
        qb = self.counter_q(team_b).sum(dim=1)
        counter = (pa * qb).sum(-1) - (pb * qa).sum(-1)  # (B,)
        logit = strength + counter
        if self.use_class_synergy:
            logit = logit + (self._class_synergy(team_a) - self._class_synergy(team_b))
        return logit

    @torch.no_grad()
    def win_prob(self, team_a, team_b, map_idx, mode_idx) -> torch.Tensor:
        return torch.sigmoid(self.forward(team_a, team_b, map_idx, mode_idx))
