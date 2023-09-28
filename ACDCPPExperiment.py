import os
import sys
sys.path.append('Automatic-Circuit-Discovery/')

from acdc.TLACDCExperiment import TLACDCExperiment
from utils.prune_utils import acdc_nodes, get_nodes, get_present_edges, ModelComponent
from utils.graphics_utils import show

from typing import Callable, List, Literal, Dict, Tuple

from transformer_lens import HookedTransformer
import torch as t
from torch import Tensor
import warnings
from tqdm import tqdm
import numpy as np
import json

class ACDCPPExperiment():

    def __init__(
        self, 
        model: HookedTransformer,
        clean_data: Tensor,
        corr_data: Tensor,
        acdc_metric: Callable[[Tensor], Tensor],
        acdcpp_metric: Callable[[Tensor], Tensor],
        acdc_thresholds: List[float],
        acdcpp_thresholds: List[float],
        run_name: str,
        save_graphs_after: float,
        verbose: bool = False,
        attr_absolute_val: bool = True,
        zero_ablation: bool = False,
        return_pruned_heads: bool = True,
        return_pruned_attr: bool = True,
        return_num_passes: bool = True,
        pass_tokens_to_metric: bool = False,
        pruning_mode: Literal["edge", "node"] = "node",
        no_pruned_nodes_attr: int = 10,
        **acdc_kwargs
    ):
        self.model = model

        self.clean_data = clean_data
        self.corr_data = corr_data

        self.run_name = run_name
        self.verbose = verbose
        
        self.acdc_metric = acdc_metric
        self.acdcpp_metric = acdcpp_metric
        self.pass_tokens_to_metric = pass_tokens_to_metric

        self.acdc_thresholds = acdc_thresholds 
        self.acdcpp_thresholds = acdcpp_thresholds 
        self.attr_absolute_val = attr_absolute_val
        self.zero_ablation = zero_ablation

        # For now, not using these (lol)
        self.return_pruned_heads = return_pruned_heads
        self.return_pruned_attr = return_pruned_attr
        self.return_num_passes = return_num_passes
        self.save_graphs_after = save_graphs_after
        self.pruning_mode: Literal["edge", "node"] = pruning_mode
        self.no_pruned_nodes_attr = no_pruned_nodes_attr

        if self.pruning_mode == "edge" and self.no_pruned_nodes_attr != 1:
            warnings.warn("I've been getting errors with no_pruned_nodes_attr > 1 with edge pruning, you may wish to switch to no_pruned_nodes_attr=1")

        self.acdc_args = acdc_kwargs
        if verbose:
            print('Set up model hooks')

    def setup_exp(self, threshold: float) -> TLACDCExperiment:
        exp = TLACDCExperiment(
            model=self.model,
            threshold=threshold,
            run_name=self.run_name,
            ds=self.clean_data,
            ref_ds=self.corr_data,
            metric=self.acdc_metric,
            zero_ablation=self.zero_ablation,
            # save_graphs_after=self.save_graphs_after,
            online_cache_cpu=False,
            corrupted_cache_cpu=False,
            verbose=self.verbose,
            **self.acdc_args
        )
        exp.model.reset_hooks()
        exp.setup_model_hooks(
            add_sender_hooks=True,
            add_receiver_hooks=True,
            doing_acdc_runs=False
        )

        return exp
    
    def run_acdcpp(self, exp: TLACDCExperiment):
        """
        Initial run of ACDCpp to calculate all attribution scores once before sweeping through thresholds
        """
        if self.verbose:
            print('Running ACDC++')
            
        for _ in range(self.no_pruned_nodes_attr):
            acdcpp_attrs = acdc_nodes(
                model=exp.model,
                clean_input=self.clean_data,
                corrupted_input=self.corr_data,
                metric=self.acdcpp_metric, 
                exp=exp,
                verbose=self.verbose,
                attr_absolute_val=self.attr_absolute_val,
                mode=self.pruning_mode,
            )
            t.cuda.empty_cache()
        return acdcpp_attrs
        
    
    def eval_acdcpp(self, exp, acdcpp_results, threshold):
        """
        Applying threshold to precalculated results from run_acdcpp()
        """

        for (parent, downstream_component), attr in acdcpp_results.items():
            if self.attr_absolute_val: 
                attr = np.abs(attr)

            # for position in exp.positions: # TODO add this back in!
            should_prune = attr < threshold
            if should_prune:
                edge_tuple = (downstream_component.hook_point_name, downstream_component.index, parent.hook_point_name, parent.index)
                exp.corr.edges[edge_tuple[0]][edge_tuple[1]][edge_tuple[2]][edge_tuple[3]].present = False
                exp.corr.remove_edge(*edge_tuple)
                if self.verbose:
                    print(f'Pruning {parent=} {downstream_component=}')

            else:
                if self.verbose: # Putting this here since tons of things get pruned when doing edges!
                    print(f'NOT PRUNING {parent=} {downstream_component=} with attribution {attr}')

        return exp

    def run_acdc(self, exp: TLACDCExperiment):
        if self.verbose:
            print('Running ACDC')
            
        while exp.current_node:
            exp.step(testing=False)

        return (get_present_edges(exp.corr), exp.num_passes)
    
    def convert_to_torch_index(index_list):
            return ''.join(['None' if i == ':' else i for i in index_list])

    def save_combined(self, present_edge_attrs, num_passes):
        cleaned_attrs = []
        for e1, i1, e2, i2, attr in present_edge_attrs.items():
            cleaned_attrs.append([e1, self.convert_to_torch_index(str(i1)), e2, self.convert_to_torch_index(str(i2)), attr])

        with open(f'res/{self.run_name}/present_edge_attrs.json', 'w') as f:
            json.dump(present_edge_attrs, f)
        with open(f'res/{self.run_name}/num_passes.json', 'w') as f:
            json.dump(num_passes, f)
    
    def save_acdcpp_edge_attributions(self, acdcpp_attr):
        # Save results
        cleaned_attrs = []
        for ((e1, i1, _), (e2, i2, _)), attr in acdcpp_attr.items():
            cleaned_attrs.append([e1, self.convert_to_torch_index(str(i1)), e2, self.convert_to_torch_index(str(i2)), attr])
                
        with open(f'res/{self.run_name}/acdcpp_only_attrs.json', 'w') as f:
            json.dump(cleaned_attrs, f)

    def run(self, save_after_acdcpp=True, save_after_acdc=True):
        # Calculate ACDC++ attributions initially, before applying thresholds in sweep
        os.makedirs(f'res/{self.run_name}', exist_ok=True)
        exp = self.setup_exp(threshold=-1) # Have to setup exp.corr.nodes for initial ACDCpp run; TODO rewrite run_acdpp run_acdcpp so it does not req an exp object explicitly
        acdcpp_attrs = self.run_acdcpp(exp)
        if save_after_acdcpp:
            self.save_acdcpp_edge_attributions(acdcpp_attrs)

        # Sweep through ACDC++ and ACDC thresholds
        os.makedirs(f'ims/{self.run_name}', exist_ok=True)
        num_passes = {}
        present_edge_attrs = {}

        for acdcpp_threshold in tqdm(self.acdcpp_thresholds, desc="ACDC++"):
            # if self.verbose:
            print(f"{acdcpp_threshold=}")
            num_passes[acdcpp_threshold] = {}
            present_edge_attrs[acdcpp_threshold] = {}

            for acdc_threshold in tqdm(self.acdc_thresholds, desc="ACDC"):
                # Setup exp for ACDC run on a subgraph pruned by ACDC++ with chosen threshold
                # if self.verbose:
                print(f"{acdc_threshold=}")
                exp = self.setup_exp(acdc_threshold)
                prepruned_exp = self.eval_acdcpp(exp, acdcpp_attrs, acdcpp_threshold)
                # Do not save acdcpp-graphs for now.
                # Only applying threshold to this one as these graphs tend to be HUGE
                # if acdcpp_threshold >= self.save_graphs_after:
                #     print('Saving ACDC++ Graph')
                #     show(exp.corr, fname=f'ims/{self.run_name}/thresh{acdcpp_threshold}_before_acdc.png')
                
                # Run ACDC on pruned subgraph
                acdc_edge_attr, passes = self.run_acdc(prepruned_exp)
                # print('Saving ACDC Graph')
                # show(prepruned_exp.corr, fname=f'ims/{self.run_name}/thresh{acdc_threshold}_after_acdc.png')
                    
                # Save results
                present_edge_attrs[acdcpp_threshold][acdc_threshold] = acdc_edge_attr
                num_passes[acdcpp_threshold][acdc_threshold] = passes
                self.save_combined(present_edge_attrs, num_passes)

                del prepruned_exp
                t.cuda.empty_cache()
            t.cuda.empty_cache()
        t.cuda.empty_cache()

        return present_edge_attrs, num_passes, acdcpp_attrs