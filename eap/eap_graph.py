import os

import numpy as np
import torch
from torch import Tensor

from functools import partial

from jaxtyping import Float
from typing import Dict

DEFAULT_GRAPH_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ims")

class EAPGraph:

    def __init__(self, cfg, upstream_nodes=None, downstream_nodes=None, edges=None):
        self.cfg = cfg

        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]

        self.valid_upstream_hook_types = ["hook_resid_pre", "hook_result", "hook_mlp_out"]
        self.valid_downstream_hook_types = ["hook_q_input", "hook_k_input", "hook_v_input", "hook_mlp_in", "hook_resid_post"]

        # TODO valid_upstream_hook_types and upstream_component_ordering can be merged into one data structure
        self.upstream_component_ordering = {
            "hook_resid_pre": 0,
            "hook_result": 1,
            "hook_mlp_out": 2,
        }
        self.downstream_component_ordering = {
            "hook_q_input": 0,
            "hook_k_input": 1,
            "hook_v_input": 2,
            "hook_mlp_in": 3,
            "hook_resid_post": 4
        }

        self.element_size = torch.empty((0), device=self.cfg.device, dtype=self.cfg.dtype).element_size()

        self.upstream_nodes = []
        self.downstream_nodes = []

        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}

        self.upstream_hook_slice: Dict[str, slice] = {}
        self.downstream_hook_slice: Dict[str, slice] = {}

        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_attn_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}

        # If a list of edges is passed we only take the nodes that are connected by these edges
        if edges is not None:
            upstream_nodes = [edge[0] for edge in edges]
            downstream_nodes = [edge[1] for edge in edges]

        self.setup_graph_from_nodes(upstream_nodes, downstream_nodes)

        # We will create these tensors when needed
        self.eap_scores: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None
        self.adj_matrix: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None

    def setup_graph_from_nodes(self, upstream_nodes=None, downstream_nodes=None):
        # if no nodes are specified, we assume that all of them will be used
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()

        # we can assume that the two lists of hooks are sorted by layer number
        self.upstream_hooks, self.downstream_hooks = self.get_hooks_from_nodes(upstream_nodes, downstream_nodes)

        upstream_node_index = 0

        for hook_name in self.upstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]
            
            # we store the slice of all upstream nodes previous to this layer
            if layer not in self.upstream_nodes_before_layer:
                # we must check previous layers too because we might have skipped some
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_nodes_before_layer:
                        self.upstream_nodes_before_layer[earlier_layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                        self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

            if hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index + 1)
                upstream_node_index += 1

            elif hook_type == "hook_result":
                for head_idx in range(self.cfg.n_heads):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_index[f"head.{layer}.{head_idx}"] = upstream_node_index + head_idx
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + self.cfg.n_heads)
                self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index + self.cfg.n_heads)
                upstream_node_index += self.cfg.n_heads 

            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                upstream_node_index += 1

            else:
                assert False, "Invalid upstream hook type"

        # if there are no more upstream nodes after a certain layer we still have
        # to save that into the slice dictionaries
        for layer in range(0, self.cfg.n_layers):
            if layer not in self.upstream_nodes_before_layer:
                self.upstream_nodes_before_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_attn_layer[layer] = slice(0, upstream_node_index)
                self.upstream_nodes_before_mlp_layer[layer] = slice(0, upstream_node_index)

        downstream_node_index = 0

        for hook_name in self.downstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if hook_type == "hook_q_input" or hook_type == "hook_k_input" or hook_type == "hook_v_input":
                letter = hook_type.split("_")[1].lower()
                for head_idx in range(self.cfg.n_heads):
                    self.downstream_nodes.append(f"head.{layer}.{head_idx}.{letter}")
                    self.downstream_node_index[f"head.{layer}.{head_idx}.{letter}"] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + self.cfg.n_heads)
                downstream_node_index += self.cfg.n_heads 

            elif hook_type == "hook_mlp_in":
                self.downstream_nodes.append(f"mlp.{layer}")
                self.downstream_node_index[f"mlp.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            elif hook_type == "hook_resid_post":
                self.downstream_nodes.append(f"resid_post.{layer}")
                self.downstream_node_index[f"resid_post.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1

            else:
                assert False, "Invalid downstream hook type"

        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)

        activations_tensor_in_gb = self.n_upstream_nodes * self.cfg.d_model * self.element_size / 2**30 
        print(f"Saving activations requires {activations_tensor_in_gb:.4f} GB of memory per token")

    # Given a set of upstream nodes and downstream nodes, this function returns the corresponding hooks
    # to access the activations of these nodes. We return the list of hooks sorted by layer number.
    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):

        # we first check that the types of the nodes passed are valid
        for node in upstream_nodes:
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"

        for node in downstream_nodes:
            node_type = node.split(".")[0] # 'resid_post', 'mlp' or 'head'
            assert node_type in self.valid_downstream_node_types, "Invalid downstream node"

        upstream_hooks = []
        downstream_hooks = []

        for node in upstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                hook_type = "hook_resid_pre" if node_type == "resid_pre" else "hook_mlp_out" if node_type == "mlp" else "attn.hook_result"
                for layer in range(self.cfg.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])
                hook_type = "hook_resid_pre" if node_type == "resid_pre" else "hook_mlp_out" if node_type == "mlp" else "attn.hook_result"
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")

        for node in downstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                if node == "head":
                    for layer in range(self.cfg.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                elif node == "resid_post" or node == "mlp":
                    hook_type = "hook_resid_post" if node == "resid_post" else "hook_mlp_in"
                    for layer in range(self.cfg.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                else:
                    raise NotImplementedError("Invalid downstream node")
            else:
                # we are in the case of a node specified for a single layer
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = "hook_resid_post" if node.startswith("resid_post") else "hook_mlp_in"
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                elif node.startswith("head"):
                    all_heads = len(node.split(".")) <= 2 # head.10 means taking all heads at layer 10
                    head_idx = None if all_heads else int(node.split(".")[2]) # we don't use this variable because we have to add the same hook whether we want one head or all
                    letters = ["q", "k", "v"]
                    if len(node.split(".")) == 4:
                        # a specific input channel is specified so we modify the hook name accordingly
                        letter_specified = node.split(".")[3]
                        assert letter_specified in letters, "Invalid letter specified"
                        letters = [letter_specified]
                    for letter in letters:
                        downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                else:
                    raise NotImplementedError("Invalid downstream node")

        upstream_hooks = list(set(upstream_hooks))
        downstream_hooks = list(set(downstream_hooks))

        def get_hook_level(hook, component_ordering):
            # Function for differentiating the order of computation in between layers, e.g. attn_layer2 is before mlp_layer2
            num_components_per_layer = len(component_ordering)
            layer = int(hook.split(".")[1])
            hook_type = hook.split(".")[-1]
            component_order = component_ordering[hook_type]
            level = layer * num_components_per_layer + component_order
            return level

        get_upstream_hook_level = partial(get_hook_level, component_ordering=self.upstream_component_ordering)
        get_downstream_hook_level = partial(get_hook_level, component_ordering=self.downstream_component_ordering)

        # we sort the hooks by the order in which they appear in the computation
        upstream_hooks = sorted(upstream_hooks, key=get_upstream_hook_level)
        downstream_hooks = sorted(downstream_hooks, key=get_downstream_hook_level)

        return upstream_hooks, downstream_hooks
    
    def get_slice_previous_upstream_nodes(self, downstream_hook):
        layer = downstream_hook.layer()
        hook_type = downstream_hook.name.split(".")[-1]
        # if hook_type == "hook_resid_post":
        #     return self.upstream_nodes_before_layer[layer + 1]
        if hook_type == "hook_mlp_in":
            return self.upstream_nodes_before_mlp_layer[layer]
        elif hook_type in ["hook_q_input", "hook_k_input", "hook_v_input", "hook_resid_post"]:
            return self.upstream_nodes_before_layer[layer]

    def get_hook_slice(self, hook_name):
        if hook_name in self.upstream_hook_slice:
            return self.upstream_hook_slice[hook_name]
        elif hook_name in self.downstream_hook_slice:
            return self.downstream_hook_slice[hook_name]

    def reset_scores(self):
        self.eap_scores = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.cfg.device
        )

    def top_edges(
        self,
        n=1000,
        threshold=None,
        abs_scores=True,
    ):
        assert self.eap_scores is not None, "EAP scores have not been computed yet"

        # get indices of maximum values in 2d tensor
        if abs_scores:
            top_scores, top_indices = torch.topk(self.eap_scores.flatten().abs(), k=n, dim=0)
        else:
            top_scores, top_indices = torch.topk(self.eap_scores.flatten(), k=n, dim=0)

        top_edges = []
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            if threshold is not None and abs_score < threshold:
                break

            upstream_node_idx, downstream_node_idx = np.unravel_index(index, self.eap_scores.shape)
            score = self.eap_scores[upstream_node_idx, downstream_node_idx]

            top_edges.append((self.upstream_nodes[upstream_node_idx], self.downstream_nodes[downstream_node_idx], score.item()))

        return top_edges

    def subgraph_top_edges(
        self,
        threshold=None,
        abs_scores=True
    ):

        assert self.eap_scores is not None, "EAP scores have not been computed yet"

        top_edges = self.top_edges(threshold=threshold, abs_scores=abs_scores)

        upstream_nodes = [edge[0] for edge in top_edges]
        downstream_nodes = [edge[1] for edge in top_edges]
        subgraph = EAPGraph(upstream_nodes, downstream_nodes)

        return subgraph

    def show(
        self,
        threshold=None,
        abs_scores=True,
        fname: str="eap_graph.png"
    ):
        import pygraphviz as pgv

        minimum_penwidth = 0.2
        edges = self.top_edges(threshold=threshold, abs_scores=abs_scores)

        g = pgv.AGraph(
            name='root',
            strict=True,
            directed=True
        )

        g.graph_attr.update(ranksep='0.1', nodesep='0.1', compound=True)
        g.node_attr.update(fixedsize='true', width='1.5', height='.5')

        def find_layer_node(node):
            if node == f'resid_post.{self.cfg.n_layers - 1}':
                return self.cfg.n_layers
            else:
                return int(node.split(".")[1])

        layer_to_subgraph = {}
        layer_to_subgraph[-1] = g.add_subgraph(name=f'cluster_-1', rank='same', color='invis')
        layer_to_subgraph[-1].add_node(f'-1_invis', style='invis')

        min_layer = 999
        max_layer = -1
        layers = list(range(0, 32))

        for edge in edges:
            parent_node = edge[0]
            child_node = edge[1]
            min_layer = min(min_layer, find_layer_node(parent_node))
            max_layer = max(max_layer, find_layer_node(child_node))

        layers = list(range(min_layer, max_layer + 1))
        prev_layer = None

        for layer in layers:
            layer_to_subgraph[layer] = g.add_subgraph(name=f'cluster_{layer}', rank='same', color='invis')
            layer_to_subgraph[layer].add_node(f'{layer}_invis', style='invis')

            if prev_layer is not None:
                g.add_edge(f'{prev_layer}_invis', f'{layer}_invis', style='invis', weight=1000)

            prev_layer = layer
                
        # Adding nodes and edges between nodes
        for edge in edges:
            parent_node, child_node, edge_score = edge

            parent_name = parent_node
            child_name = child_node

            child_name = child_name.replace(".q", "").replace(".k", "").replace(".v", "")
            
            for node_name in [parent_name, child_name]:

                node_layer = find_layer_node(node_name)

                node_color = '#1f77b4' if node_name.startswith("head") else '#ff7f0e' if node_name.startswith("mlp") else '#2ca02c' if node_name.startswith("resid") else '#d62728'

                layer_to_subgraph[node_layer].add_node(
                    node_name,
                    fillcolor=node_color,
                    color="black",
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )
                
            edge_width = str(max(minimum_penwidth, edge_score*100))

            g.add_edge(
                parent_name,
                child_name,
                penwidth=edge_width,
                color='#0091E4',
                weight=10,
                minlen='0.5',
            )

        save_path = os.path.join(DEFAULT_GRAPH_PLOT_DIR, fname)

        print(f"Saving graph")
        if not fname.endswith(".gv"): # turn the .gv file into a .png file
            g.draw(path=save_path, prog='dot')

        return g
