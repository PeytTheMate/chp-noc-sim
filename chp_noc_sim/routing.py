from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, List, Mapping, Optional, Sequence, Tuple, TypeVar
import heapq

from .config import PhysicsConfig
from .entities import Link, Node

NodeIdT = TypeVar("NodeIdT", bound=Hashable)


@dataclass
class RoutingAgent(Generic[NodeIdT]):
    """
    Naive deterministic routing agent.

    It assumes *mean* physics only (no variance) and performs a shortest-path
    search minimizing expected dB loss. It does *not* see Monte Carlo noise.
    """
    nodes: Mapping[NodeIdT, Node]
    adjacency: Mapping[NodeIdT, Sequence[Link]]
    physics_config: PhysicsConfig

    def _link_cost_db(self, link: Link) -> float:
        """
        Deterministic cost for a link, used internally by the routing agent.

        Here we approximate each bent link as containing a single 90° turn so
        that BENDING_LOSS_DB_PER_TURN is reflected in the path metric.
        """
        bends = 1 if link.is_bent else 0
        return (
            link.length_microns * self.physics_config.PROPAGATION_LOSS_MEAN_DB_PER_MICRON
            + bends * self.physics_config.BENDING_LOSS_DB_PER_TURN
        )

    def plan_path(
        self,
        source_id: NodeIdT,
        dest_id: NodeIdT,
    ) -> List[Link]:
        """
        Compute a simple shortest-loss path from source to destination.

        Returns:
            A list of Links in traversal order. If no path satisfies the
            global LINK_BUDGET_DB constraint (under mean physics), returns [].
        """
        if source_id == dest_id:
            return []

        # Dijkstra on (node) with cost = cumulative deterministic loss.
        INF = float("inf")
        dist: Dict[NodeIdT, float] = {node_id: INF for node_id in self.nodes}
        prev: Dict[NodeIdT, Optional[Tuple[NodeIdT, Link]]] = {
            node_id: None for node_id in self.nodes
        }

        dist[source_id] = 0.0
        heap: List[Tuple[float, NodeIdT]] = [(0.0, source_id)]

        while heap:
            cost_so_far, node_id = heapq.heappop(heap)
            if cost_so_far > dist[node_id]:
                continue
            if cost_so_far > self.physics_config.LINK_BUDGET_DB:
                # Further exploration will only increase loss; prune.
                continue
            if node_id == dest_id:
                break

            for link in self.adjacency.get(node_id, ()):
                neighbor_id = link.dst.id  # Node IDs and Node.id must be aligned
                step_cost = self._link_cost_db(link)
                new_cost = cost_so_far + step_cost
                if new_cost < dist[neighbor_id] and new_cost <= self.physics_config.LINK_BUDGET_DB:
                    dist[neighbor_id] = new_cost
                    prev[neighbor_id] = (node_id, link)
                    heapq.heappush(heap, (new_cost, neighbor_id))

        if dist[dest_id] == INF:
            # No feasible path within the link budget (under mean physics).
            return []

        # Reconstruct path.
        path_links: List[Link] = []
        cur: Optional[Tuple[NodeIdT, Link]]
        node_id = dest_id
        while node_id != source_id:
            cur = prev[node_id]
            if cur is None:
                # No path actually stored (should not happen if dist[dest_id] != INF).
                return []
            parent_id, link = cur
            path_links.append(link)
            node_id = parent_id

        path_links.reverse()
        return path_links
