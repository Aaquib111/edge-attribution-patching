import sys
sys.path.append('Automatic-Circuit-Discovery/')

from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import EdgeType
import numpy as np

import pygraphviz as pgv
from pathlib import Path

import tqdm.notebook as tqdm

from typing import Union, Dict, Optional

def replace_parens(tup):
    '''
        Method to standardize the TorchIndex of components to a string
    '''
    return str(tup).replace('(', '[').replace(')', ']').replace('[None,]', '[None]').replace('None', ':')

def get_node_name(node: TLACDCInterpNode, show_full_index=True):
    """Node name for use in pretty graphs"""

    if not show_full_index:
        name = ""
        qkv_substrings = [f"hook_{letter}" for letter in ["q", "k", "v"]]
        qkv_input_substrings = [f"hook_{letter}_input" for letter in ["q", "k", "v"]]

        # Handle embedz
        if "resid_pre" in node.name:
            assert "0" in node.name and not any([str(i) in node.name for i in range(1, 10)])
            name += "embed"
            if len(node.index.hashable_tuple) > 2:
                name += f"_[{node.index.hashable_tuple[2]}]"
            return name

        elif "embed" in node.name:
            name = "pos_embeds" if "pos" in node.name else "token_embeds"

        # Handle q_input and hook_q etc
        elif any([node.name.endswith(qkv_input_substring) for qkv_input_substring in qkv_input_substrings]):
            relevant_letter = None
            for letter, qkv_substring in zip(["q", "k", "v"], qkv_substrings):
                if qkv_substring in node.name:
                    assert relevant_letter is None
                    relevant_letter = letter
            name += "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2]) + "_" + relevant_letter

        # Handle attention hook_result
        elif "hook_result" in node.name or any([qkv_substring in node.name for qkv_substring in qkv_substrings]):
            name = "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2])

        # Handle MLPs
        elif node.name.endswith("resid_mid"):
            raise ValueError("We removed resid_mid annotations. Call these mlp_in now.")
        elif node.name.endswith("mlp_out") or node.name.endswith("mlp_in"):
            name = "m" + node.name.split(".")[1]

        # Handle resid_post
        elif "resid_post" in node.name:
            name += "resid_post"

        else:
            raise ValueError(f"Unrecognized node name {node.name}")

    else:
        
        name = node.name + str(node.index.graphviz_index(use_actual_colon=True))

    return "<" + name + ">"

def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex((np.random.randint(120, 256), np.random.randint(120, 256), np.random.randint(120, 256)))

def build_colorscheme(correspondence, colorscheme: str = "Pastel2", show_full_index=True) -> Dict[str, str]:
    colors = {}
    for node in correspondence.nodes():
        colors[get_node_name(node, show_full_index=show_full_index)] = generate_random_color(colorscheme)
    return colors

def get_node_color(node_name):
    if '<a' in node_name:
        # Attention head
        return '#1f77b4'
    elif '<m' in node_name:
        return '#ff7f0e'
    else:
        return '#7f7f7f'
    
def get_edge_props(edge_name, edge_to_attr):
    if edge_to_attr[edge_name] < 0:
        edge_color = '#d62728'
    else:
        edge_color = '#2ca02c'

    edge_width = abs(edge_to_attr[edge_name])
    
    return edge_width, edge_color

def show(
    correspondence: TLACDCInterpNode,
    edge_to_attr: dict = None,
    fname=None,
    colorscheme: Union[Dict, str] = "Pastel2",
    minimum_penwidth: float = 0.3,
    show_full_index: bool = False,
    remove_self_loops: bool = True,
    remove_qkv: bool = True,
    layout: str="dot",
    edge_type_colouring: bool = False,
    show_placeholders: bool = False,
    seed: Optional[int] = None
):
    g = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout=layout)

    if seed is not None:
        np.random.seed(seed)
    
    groups = {}
    if isinstance(colorscheme, str):
        colors = build_colorscheme(correspondence, colorscheme, show_full_index=show_full_index)
    else:
        colors = colorscheme
        for name, color in colors.items():
            if color not in groups:
                groups[color] = [name]
            else:
                groups[color].append(name)

    node_pos = {}
    if fname is not None:
        base_fname = ".".join(str(fname).split(".")[:-1])

        base_path = Path(base_fname)
        fpath = base_path / "layout.gv"
        if fpath.exists():
            g_pos = pgv.AGraph()
            g_pos.read(fpath)
            for node in g_pos.nodes():
                node_pos[node.name] = node.attr["pos"]
    
    for child_hook_name in correspondence.edges:
        for child_index in correspondence.edges[child_hook_name]:
            for parent_hook_name in correspondence.edges[child_hook_name][child_index]:
                for parent_index in correspondence.edges[child_hook_name][child_index][parent_hook_name]:
                    edge = correspondence.edges[child_hook_name][child_index][parent_hook_name][parent_index]
                    edge_name=f'{child_hook_name}{replace_parens(child_index)}{parent_hook_name}{replace_parens(parent_index)}'
                    parent = correspondence.graph[parent_hook_name][parent_index]
                    child = correspondence.graph[child_hook_name][child_index]
                    
                    parent_name = get_node_name(parent, show_full_index=show_full_index)
                    child_name = get_node_name(child, show_full_index=show_full_index)
                    
                    if remove_qkv:
                        if any(qkv in child_name or qkv in parent_name for qkv in ['_q_', '_k_', '_v_']):
                            continue
                        parent_name = parent_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
                        child_name = child_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
                        
                    if remove_self_loops and parent_name == child_name:
                        # Important this go after the qkv removal
                        continue
                    
                    if edge.present and (edge.edge_type != EdgeType.PLACEHOLDER or show_placeholders):
                        #print(f'Edge from {parent_name=} to {child_name=}')
                        for node_name in [parent_name, child_name]:
                            maybe_pos = {}
                            if node_name in node_pos:
                                maybe_pos["pos"] = node_pos[node_name]
                            g.add_node(
                                node_name,
                                fillcolor=get_node_color(node_name),
                                color="black",
                                style="filled, rounded",
                                shape="box",
                                fontname="Helvetica",
                                **maybe_pos,
                            )
                        
                        if not edge_to_attr or not edge_name in edge_to_attr:
                            edge_width = minimum_penwidth * 2
                            edge_color = get_node_color(parent_name)
                        else:
                            edge_width, edge_color = get_edge_props(edge_name, edge_to_attr)
                            
                        g.add_edge(
                            parent_name,
                            child_name,
                            penwidth=edge_width,
                            color=edge_color,
                        )
    if fname is not None:
        base_fname = ".".join(str(fname).split(".")[:-1])

        base_path = Path(base_fname)
        base_path.mkdir(exist_ok=True)
        for k, s in groups.items():
            g2 = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout="neato")
            for node_name in s:
                g2.add_node(
                    node_name,
                    style="filled, rounded",
                    shape="box",
                )
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    g2.add_edge(s[i], s[j], style="invis", weight=200)
            g2.write(path=base_path / f"{k}.gv")

        g.write(path=base_fname + ".gv")

        if not fname.endswith(".gv"): # turn the .gv file into a .png file
            g.draw(path=fname, prog="dot")

    return g