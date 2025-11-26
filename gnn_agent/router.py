from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data

from chp_noc_sim import NetworkTopology, PhysicsConfig
from .model import RiskAwareGATConfig, RiskAwareGATPolicy


@dataclass
class GNNRouter:
    """
    Thin adapter around RiskAwareGATPolicy for use inside the Phase 1 simulator.

    It builds a PyG graph from the current NetworkTopology and returns
    a chosen next-hop neighbor for the current node.
    """

    physics_config: PhysicsConfig
    model: RiskAwareGATPolicy
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_data_from_topology(
        topology: NetworkTopology,
        physics_cfg: PhysicsConfig,
    ) -> Tuple[Data, Dict[str, int], List[str]]:
        """
        Convert NetworkTopology -> PyG Data and provide ID↔index mappings.

        Node features: [queue_length, temperature_k]
        Edge features: [length_microns, estimated_loss_variance]
        """
        node_ids = sorted(topology.nodes.keys())
        node_id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        x = torch.zeros((num_nodes, 2), dtype=torch.float32)
        for nid, idx in node_id_to_idx.items():
            node = topology.nodes[nid]
            queue_len = float(getattr(node, "queue_length", 0.0))
            temp_k = float(node.temperature_k)
            x[idx, 0] = queue_len
            x[idx, 1] = temp_k

        src_list: List[int] = []
        dst_list: List[int] = []
        edge_feat_list: List[List[float]] = []

        base_var_per_micron = physics_cfg.PROPAGATION_LOSS_STD_DB_PER_MICRON ** 2

        for link in topology.links:
            src_idx = node_id_to_idx[link.src.id]
            dst_idx = node_id_to_idx[link.dst.id]

            length = float(link.length_microns)
            var = base_var_per_micron * length

            src_list.append(src_idx)
            dst_list.append(dst_idx)
            edge_feat_list.append([length, var])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_nodes
        return data, node_id_to_idx, node_ids

    def get_next_hop(
        self,
        current_node: str,
        destination_node: str,
        network_state: NetworkTopology,
    ) -> str:
        """
        Select a next-hop neighbor using the GNN policy.

        Args:
            current_node: Node ID (string) of the packet's current router.
            destination_node: Node ID (string) of the packet's target.
            network_state: Current network topology (Phase 1 view).

        Returns:
            next_node_id: ID of the chosen neighbor. If no valid neighbor
            exists, returns current_node (i.e., "stay").
        """
        data, id_to_idx, node_ids = self._build_data_from_topology(
            topology=network_state,
            physics_cfg=self.physics_config,
        )

        if current_node not in id_to_idx or destination_node not in id_to_idx:
            return current_node

        src_idx = id_to_idx[current_node]
        dest_idx = id_to_idx[destination_node]

        data = data.to(self.device)

        with torch.no_grad():
            edge_logits: Tensor = self.model(data, dest_index=dest_idx)

        edge_index = data.edge_index
        mask = edge_index[0] == src_idx
        candidate_idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if candidate_idxs.numel() == 0:
            # No outgoing edges from current_node.
            return current_node

        candidate_logits = edge_logits[candidate_idxs]
        probs = F.softmax(candidate_logits, dim=0)

        # Deterministic argmax next-hop (you can change to sampling if desired).
        best_idx = int(torch.argmax(probs).item())
        chosen_edge_idx = int(candidate_idxs[best_idx].item())
        next_node_idx = int(edge_index[1, chosen_edge_idx].item())

        idx_to_id = {idx: nid for nid, idx in id_to_idx.items()}
        return idx_to_id[next_node_idx]

@dataclass
class GNNRoutingAdapter:
    """
    Adapter that makes a GNNRouter look like a RoutingAgent:
    it exposes plan_path(source_id, dest_id) -> List[Link].

    This lets you drop it into NetworkSimulation without changing the sim.
    """
    topology: NetworkTopology
    gnn_router: GNNRouter
    max_hop_factor: int = 4

    def plan_path(self, source_id: str, dest_id: str) -> List[Link]:
        if source_id == dest_id:
            return []

        current_id = source_id
        visited: set[str] = {source_id}
        max_hops = len(self.topology.nodes) * self.max_hop_factor
        path: List[Link] = []

        for _ in range(max_hops):
            if current_id == dest_id:
                break

            next_id = self.gnn_router.get_next_hop(
                current_node=current_id,
                destination_node=dest_id,
                network_state=self.topology,
            )

            # No progress or loop → fail.
            if next_id == current_id or next_id in visited:
                return []

            outgoing = self.topology.adjacency.get(current_id, [])
            candidates = [lnk for lnk in outgoing if lnk.dst.id == next_id]
            if not candidates:
                return []

            link = candidates[0]
            path.append(link)
            visited.add(next_id)
            current_id = next_id

        if current_id != dest_id:
            return []
        return path

def build_default_gnn_router(physics_cfg: PhysicsConfig) -> GNNRouter:
    """
    Convenience builder: create a GNNRouter with a fresh lightweight model.
    Typically you would load trained weights here.
    """
    cfg = RiskAwareGATConfig()
    model = RiskAwareGATPolicy(cfg)
    router = GNNRouter(physics_config=physics_cfg, model=model)
    return router
