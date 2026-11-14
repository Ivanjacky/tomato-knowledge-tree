from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ModalityGate(nn.Module):
    """Projects text/image vectors and fuses only the available modalities."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        fusion_mode: str = "gate",
        context_residual_scale: float = 0.35,
        context_gate_floor: float = 0.0,
        text_modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"gate", "mean", "context_gate"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if context_residual_scale < 0:
            raise ValueError("context_residual_scale must be non-negative")
        if not 0.0 <= context_gate_floor < 1.0:
            raise ValueError("context_gate_floor must be in [0, 1)")
        if not 0.0 <= text_modality_dropout < 1.0:
            raise ValueError("text_modality_dropout must be in [0, 1)")
        self.fusion_mode = fusion_mode
        self.context_residual_scale = float(context_residual_scale)
        self.context_gate_floor = float(context_gate_floor)
        self.text_modality_dropout = float(text_modality_dropout)
        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if self.fusion_mode == "context_gate":
            # The safe camera/pose text is acquisition context rather than BBCH
            # content.  It therefore modulates the visual branch instead of
            # replacing it through a convex text/image average.
            self.context_adapter = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            self.context_norm = nn.LayerNorm(hidden_dim)
            self.context_strength_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            # Preserve the established gated-fusion path at initialization. The
            # context branch starts as an exact zero residual, then learns only
            # from parent-placement and counterfactual objectives.
            nn.init.zeros_(self.context_adapter[3].weight)
            nn.init.zeros_(self.context_adapter[3].bias)
            nn.init.constant_(self.context_strength_gate[2].bias, math.log(0.10 / 0.90))
        else:
            self.context_adapter = None
            self.context_norm = None
            self.context_strength_gate = None
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.missing = nn.Parameter(torch.zeros(hidden_dim))

    def fuse_with_diagnostics(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
        image_hidden_override: torch.Tensor | None = None,
        apply_modality_dropout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_hidden = F.normalize(self.text_projection(text), dim=-1)
        image_hidden = F.normalize(
            self.image_projection(image)
            if image_hidden_override is None
            else image_hidden_override,
            dim=-1,
        )
        text_mask = text_mask.float().unsqueeze(-1)
        image_mask = image_mask.float().unsqueeze(-1)
        if self.training and apply_modality_dropout and self.text_modality_dropout > 0.0:
            both = (text_mask > 0) & (image_mask > 0)
            keep = torch.rand_like(text_mask) >= self.text_modality_dropout
            text_mask = text_mask * torch.where(both, keep, torch.ones_like(keep))

        if self.fusion_mode in {"gate", "context_gate"}:
            gate = self.gate(torch.cat([text_hidden, image_hidden, text_mask, image_mask], dim=-1))
        else:
            gate = torch.full_like(text_mask, 0.5)

        if self.fusion_mode == "context_gate":
            assert self.context_adapter is not None
            assert self.context_norm is not None
            assert self.context_strength_gate is not None
            adapter_input = torch.cat(
                [
                    image_hidden,
                    text_hidden,
                    image_hidden * text_hidden,
                    torch.abs(image_hidden - text_hidden),
                ],
                dim=-1,
            )
            gamma, beta = self.context_adapter(adapter_input).chunk(2, dim=-1)
            scale = self.context_residual_scale
            context_delta = self.context_norm(
                scale * (image_hidden * torch.tanh(gamma) + torch.tanh(beta))
            )
            text_weight = gate * text_mask
            image_weight = (1.0 - gate) * image_mask
            denominator = text_weight + image_weight
            base_fused = (
                text_weight * text_hidden + image_weight * image_hidden
            ) / denominator.clamp_min(1e-8)
            learned_context_gate = self.context_strength_gate(
                torch.cat([adapter_input, text_mask, image_mask], dim=-1)
            )
            context_gate = (
                self.context_gate_floor + (1.0 - self.context_gate_floor) * learned_context_gate
            )
            both = text_mask * image_mask
            fused = base_fused + both * context_gate * context_delta
            effective_gate = context_gate * both
        else:
            text_weight = gate * text_mask
            image_weight = (1.0 - gate) * image_mask
            denominator = text_weight + image_weight
            fused = (
                text_weight * text_hidden + image_weight * image_hidden
            ) / denominator.clamp_min(1e-8)
            effective_gate = gate * text_mask * image_mask
        neither = (denominator <= 1e-8).float()
        fused = fused * (1.0 - neither) + self.missing.unsqueeze(0) * neither
        return self.output_norm(fused), text_hidden, image_hidden, effective_gate

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
        image_hidden_override: torch.Tensor | None = None,
        apply_modality_dropout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, text_hidden, image_hidden, _ = self.fuse_with_diagnostics(
            text,
            image,
            text_mask,
            image_mask,
            image_hidden_override=image_hidden_override,
            apply_modality_dropout=apply_modality_dropout,
        )
        return fused, text_hidden, image_hidden


class PoincareBall:
    def __init__(self, curvature: float = 1.0, eps: float = 1e-5) -> None:
        if curvature <= 0:
            raise ValueError("Curvature must be positive")
        self.curvature = float(curvature)
        self.eps = float(eps)

    def project(self, value: torch.Tensor) -> torch.Tensor:
        max_norm = (1.0 - self.eps) / math.sqrt(self.curvature)
        norm = value.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return value * scale

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        sqrt_c = math.sqrt(self.curvature)
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        mapped = torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm)
        return self.project(mapped)

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_sq = (first * first).sum(dim=-1)
        second_sq = (second * second).sum(dim=-1)
        difference_sq = ((first - second) ** 2).sum(dim=-1)
        denominator = (
            (1.0 - self.curvature * first_sq) * (1.0 - self.curvature * second_sq)
        ).clamp_min(self.eps)
        argument = 1.0 + 2.0 * self.curvature * difference_sq / denominator
        return torch.acosh(argument.clamp_min(1.0 + self.eps)) / math.sqrt(self.curvature)

    def origin_distance(self, value: torch.Tensor) -> torch.Tensor:
        sqrt_c = math.sqrt(self.curvature)
        scaled_norm = (sqrt_c * value.norm(dim=-1)).clamp(max=1.0 - self.eps)
        return 2.0 * torch.atanh(scaled_norm) / sqrt_c


class HierarchicalTreeModel(nn.Module):
    """Multimodal parent predictor with a directed hyperbolic hierarchy score."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        tree_dim: int,
        dropout: float = 0.1,
        curvature: float = 1.0,
        fusion_mode: str = "gate",
        geometry: str = "poincare",
        score_mode: str = "directed",
        controlled_tangent: bool = False,
        max_tangent_norm: float = 1.5,
        level_aware_radius: bool = False,
        max_level: int = 4,
        min_tangent_radius: float = 0.10,
        max_tangent_radius: float = 1.20,
        radius_residual_scale: float = 0.05,
        use_group_context: bool = False,
        group_context_mode: str = "residual",
        use_prototype_head: bool = False,
        use_geometric_prototypes: bool = False,
        candidate_count: int | None = None,
        relation_dim: int = 256,
        geometric_residual_scale: float = 0.20,
        distance_scale_init: float = 0.70,
        context_residual_scale: float = 0.35,
        context_gate_floor: float = 0.0,
        text_modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if geometry not in {"poincare", "euclidean"}:
            raise ValueError(f"Unsupported geometry: {geometry}")
        if score_mode not in {"directed", "symmetric_distance"}:
            raise ValueError(f"Unsupported score mode: {score_mode}")
        if max_tangent_norm <= 0:
            raise ValueError("max_tangent_norm must be positive")
        if level_aware_radius and not 0 < min_tangent_radius < max_tangent_radius:
            raise ValueError("Level-aware tangent radii must satisfy 0 < min < max")
        if radius_residual_scale < 0:
            raise ValueError("radius_residual_scale must be non-negative")
        if use_prototype_head and (candidate_count is None or candidate_count <= 0):
            raise ValueError("Prototype scoring requires a positive candidate_count")
        if group_context_mode not in {"residual", "concat"}:
            raise ValueError(f"Unsupported group context mode: {group_context_mode}")
        if use_geometric_prototypes and (candidate_count is None or candidate_count <= 0):
            raise ValueError("Geometric prototypes require a positive candidate_count")
        if geometric_residual_scale < 0:
            raise ValueError("geometric_residual_scale must be non-negative")
        if distance_scale_init <= 0:
            raise ValueError("distance_scale_init must be positive")
        self.geometry = geometry
        self.score_mode = score_mode
        self.controlled_tangent = bool(controlled_tangent)
        self.max_tangent_norm = float(max_tangent_norm)
        self.level_aware_radius = bool(level_aware_radius)
        self.max_level = int(max_level)
        self.radius_residual_scale = float(radius_residual_scale)
        self.use_group_context = bool(use_group_context)
        self.group_context_mode = str(group_context_mode)
        self.use_prototype_head = bool(use_prototype_head)
        self.use_geometric_prototypes = bool(use_geometric_prototypes)
        self.geometric_residual_scale = float(geometric_residual_scale)
        self.fusion = ModalityGate(
            embedding_dim,
            hidden_dim,
            dropout,
            fusion_mode=fusion_mode,
            context_residual_scale=context_residual_scale,
            context_gate_floor=context_gate_floor,
            text_modality_dropout=text_modality_dropout,
        )
        if self.use_group_context and self.group_context_mode == "residual":
            self.group_projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            self.delta_projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            self.context_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.context_norm = nn.LayerNorm(hidden_dim)
            self.context_image_projection = None
        elif self.use_group_context:
            self.context_image_projection = nn.Sequential(
                nn.LayerNorm(embedding_dim * 3),
                nn.Linear(embedding_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.group_projection = None
            self.delta_projection = None
            self.context_gate = None
            self.context_norm = None
        else:
            self.context_image_projection = None
            self.group_projection = None
            self.delta_projection = None
            self.context_gate = None
            self.context_norm = None
        self.tree_projection = nn.Sequential(
            nn.Linear(hidden_dim, tree_dim),
            nn.Tanh(),
        )
        self.radius_head = (
            nn.Linear(hidden_dim, 1) if self.controlled_tangent or self.level_aware_radius else None
        )
        if self.level_aware_radius:
            level_radii = torch.linspace(
                min_tangent_radius,
                max_tangent_radius,
                self.max_level + 1,
                dtype=torch.float32,
            )
            self.register_buffer("level_tangent_radii", level_radii)
        else:
            self.level_tangent_radii = None
        self.direction_score = (
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if score_mode == "directed"
            else None
        )
        inverse_softplus = math.log(math.expm1(distance_scale_init))
        self.distance_scale_raw = nn.Parameter(torch.tensor(inverse_softplus))
        self.candidate_count = (
            int(candidate_count)
            if candidate_count is not None
            and (self.use_geometric_prototypes or self.use_prototype_head)
            else None
        )
        if self.use_geometric_prototypes:
            assert candidate_count is not None
            self.geometric_parent_residual = nn.Parameter(
                torch.empty(int(candidate_count), tree_dim)
            )
            nn.init.normal_(self.geometric_parent_residual, mean=0.0, std=0.01)
            self.geometric_parent_bias = nn.Parameter(torch.zeros(int(candidate_count)))
        else:
            self.geometric_parent_residual = None
            self.geometric_parent_bias = None
        if self.use_prototype_head:
            assert candidate_count is not None
            self.parent_key = nn.Linear(hidden_dim, relation_dim, bias=False)
            self.child_query = nn.Linear(hidden_dim, relation_dim, bias=False)
            self.parent_residual = nn.Parameter(torch.empty(self.candidate_count, relation_dim))
            nn.init.normal_(self.parent_residual, mean=0.0, std=0.01)
            self.parent_bias = nn.Parameter(torch.zeros(self.candidate_count))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.direction_scale_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.parent_key = None
            self.child_query = None
            self.parent_residual = None
            self.parent_bias = None
            self.logit_scale = None
            self.direction_scale_raw = None
        self.ball = PoincareBall(curvature=curvature)

    def encode(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
        group_image: torch.Tensor | None = None,
        image_delta: torch.Tensor | None = None,
        levels: torch.Tensor | None = None,
        apply_modality_dropout: bool = True,
    ) -> dict[str, torch.Tensor]:
        context_image_hidden = None
        if self.use_group_context and self.group_context_mode == "concat":
            if group_image is None or image_delta is None:
                raise ValueError("Group-context model requires group_image and image_delta")
            assert self.context_image_projection is not None
            context_image_hidden = self.context_image_projection(
                torch.cat([image, group_image, image_delta], dim=-1)
            )
        fused, text_hidden, image_hidden, fusion_gate = self.fusion.fuse_with_diagnostics(
            text,
            image,
            text_mask,
            image_mask,
            image_hidden_override=context_image_hidden,
            apply_modality_dropout=apply_modality_dropout,
        )
        if self.use_group_context and self.group_context_mode == "residual":
            if group_image is None or image_delta is None:
                raise ValueError("Group-context model requires group_image and image_delta")
            assert self.group_projection is not None
            assert self.delta_projection is not None
            assert self.context_gate is not None
            assert self.context_norm is not None
            group_hidden = F.normalize(self.group_projection(group_image), dim=-1)
            delta_hidden = F.normalize(self.delta_projection(image_delta), dim=-1)
            context_gate = self.context_gate(torch.cat([fused, group_hidden, delta_hidden], dim=-1))
            context = 0.5 * (group_hidden + delta_hidden)
            available = image_mask.float().unsqueeze(-1)
            fused = self.context_norm(fused + available * context_gate * context)

        projected = self.tree_projection(fused)
        if self.level_aware_radius:
            if levels is None:
                raise ValueError("Level-aware radius requires node levels")
            if torch.any(levels < 0) or torch.any(levels > self.max_level):
                raise ValueError("Node level is outside the configured hierarchy")
            assert self.radius_head is not None
            assert self.level_tangent_radii is not None
            direction = F.normalize(projected, dim=-1)
            base_radius = self.level_tangent_radii[levels.long()].unsqueeze(-1)
            residual = self.radius_residual_scale * torch.tanh(self.radius_head(fused))
            tangent = direction * (base_radius + residual).clamp_min(1e-4)
        elif self.geometry == "poincare" and self.controlled_tangent:
            assert self.radius_head is not None
            direction = F.normalize(projected, dim=-1)
            tangent_norm = self.max_tangent_norm * torch.sigmoid(self.radius_head(fused))
            tangent = direction * tangent_norm
        else:
            tangent = projected
        tree_coordinates = self.ball.expmap0(tangent) if self.geometry == "poincare" else tangent
        return {
            "fused": fused,
            "text": text_hidden,
            "image": image_hidden,
            "fusion_gate": fusion_gate,
            "tangent": tangent,
            # The stable key keeps the training interface backward compatible;
            # Euclidean ablations store their 128-D tree coordinates here directly.
            "hyperbolic": tree_coordinates,
        }

    def score_candidates(
        self,
        parent_fused: torch.Tensor,
        parent_hyperbolic: torch.Tensor,
        child_fused: torch.Tensor,
        child_hyperbolic: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, parent_count = child_fused.shape[0], parent_fused.shape[0]
        parents = parent_fused.unsqueeze(0).expand(batch_size, parent_count, -1)
        children = child_fused.unsqueeze(1).expand(batch_size, parent_count, -1)
        if self.direction_score is None:
            directed = torch.zeros(
                (batch_size, parent_count), dtype=parents.dtype, device=parents.device
            )
        else:
            features = torch.cat(
                [parents, children, parents * children, torch.abs(parents - children)], dim=-1
            )
            directed = self.direction_score(features).squeeze(-1)
        parent_h = parent_hyperbolic.unsqueeze(0).expand(batch_size, parent_count, -1)
        child_h = child_hyperbolic.unsqueeze(1).expand(batch_size, parent_count, -1)
        if self.use_geometric_prototypes:
            if parent_count != self.candidate_count:
                raise ValueError(
                    f"Expected {self.candidate_count} candidate parents, received {parent_count}"
                )
            assert self.geometric_parent_residual is not None
            parent_norm = parent_hyperbolic.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            refined_direction = F.normalize(
                parent_hyperbolic + self.geometric_residual_scale * self.geometric_parent_residual,
                dim=-1,
            )
            refined_parent = refined_direction * parent_norm
            parent_h = refined_parent.unsqueeze(0).expand(batch_size, parent_count, -1)
        if self.geometry == "poincare":
            distance = self.ball.distance(parent_h, child_h)
        else:
            distance = torch.linalg.vector_norm(parent_h - child_h, dim=-1)
        distance_scale = F.softplus(self.distance_scale_raw) + 1e-4
        geometric_bias = 0.0
        if self.use_geometric_prototypes:
            assert self.geometric_parent_bias is not None
            geometric_bias = self.geometric_parent_bias.unsqueeze(0)
        if not self.use_prototype_head:
            return directed - distance_scale * distance + geometric_bias

        if parent_count != self.candidate_count:
            raise ValueError(
                f"Expected {self.candidate_count} candidate parents, received {parent_count}"
            )
        assert self.parent_key is not None
        assert self.child_query is not None
        assert self.parent_residual is not None
        assert self.parent_bias is not None
        assert self.logit_scale is not None
        assert self.direction_scale_raw is not None
        keys = F.normalize(self.parent_key(parent_fused) + self.parent_residual, dim=-1)
        queries = F.normalize(self.child_query(child_fused), dim=-1)
        prototype = self.logit_scale.exp().clamp(max=100.0) * (queries @ keys.transpose(0, 1))
        direction_scale = F.softplus(self.direction_scale_raw)
        return (
            prototype
            + direction_scale * directed
            - distance_scale * distance
            + geometric_bias
            + self.parent_bias.unsqueeze(0)
        )

    def depth_loss(
        self, parent_hyperbolic: torch.Tensor, child_hyperbolic: torch.Tensor, margin: float
    ) -> torch.Tensor:
        if self.geometry == "poincare":
            parent_radius = self.ball.origin_distance(parent_hyperbolic)
            child_radius = self.ball.origin_distance(child_hyperbolic)
        else:
            parent_radius = parent_hyperbolic.norm(dim=-1)
            child_radius = child_hyperbolic.norm(dim=-1)
        return F.relu(parent_radius - child_radius + margin).mean()

    @staticmethod
    def alignment_loss(
        text_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        both = (text_mask > 0) & (image_mask > 0)
        if not torch.any(both):
            return text_hidden.sum() * 0.0
        similarity = F.cosine_similarity(text_hidden[both], image_hidden[both], dim=-1)
        return (1.0 - similarity).mean()
