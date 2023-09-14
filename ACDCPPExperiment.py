import os
import sys
sys.path.append('Automatic-Circuit-Discovery/')

from acdc.TLACDCExperiment import TLACDCExperiment
from utils.prune_utils import acdc_nodes, get_nodes
from utils.graphics_utils import show

from typing import Callable, List

from transformer_lens import HookedTransformer
import torch as t
from torch import Tensor

class ACDCPPExperiment():

    def __init__(
        self, 
        model: HookedTransformer,
        clean_data: Tensor,
        corr_data: Tensor,
        acdc_metric: Callable[[Tensor], Tensor],
        acdcpp_metric: Callable[[Tensor], Tensor],
        thresholds: List[float],
        run_name: str,
        verbose: bool = False,
        attr_absolute_val: bool = True,
        zero_ablation: bool = False,
        return_pruned_heads: bool = True,
        return_pruned_attr: bool = True,
        return_num_passes: bool = True,
        **acdc_kwargs
    ):
        self.model = model

        self.clean_data = clean_data
        self.corr_data = corr_data

        self.run_name = run_name

        self.acdc_metric = acdc_metric
        self.acdcpp_metric = acdcpp_metric

        self.thresholds = thresholds 
        self.attr_absolute_val = attr_absolute_val
        self.zero_ablation = zero_ablation

        # For now, not using these (lol)
        self.return_pruned_heads = return_pruned_heads
        self.return_pruned_attr = return_pruned_attr
        self.return_num_passes = return_num_passes

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
            **self.acdc_args
        )
        exp.model.reset_hooks()
        exp.setup_model_hooks(
            add_sender_hooks=True,
            add_receiver_hooks=True,
            doing_acdc_runs=False
        )

        return exp
    
    def run_acdcpp(self, exp: TLACDCExperiment, threshold: float):
        for _ in range(10):
            pruned_nodes_attr = acdc_nodes(
                model=exp.model,
                clean_input=self.clean_data,
                corrupted_input=self.corr_data,
                metric=self.acdcpp_metric, 
                threshold=threshold,
                exp=exp,
                attr_absolute_val=self.attr_absolute_val
            )
            t.cuda.empty_cache()
        return (get_nodes(exp.corr), pruned_nodes_attr)

    def run_acdc(self, exp: TLACDCExperiment):
        while exp.current_node:
            exp.step(testing=False)

        return (get_nodes(exp.corr), exp.num_passes)

    def run(self, save_after_acdcpp=True, save_after_acdc=True):
        os.makedirs(f'ims/{self.run_name}', exist_ok=True)

        pruned_heads = {}
        num_passes = {}
        pruned_attrs = {}

        for threshold in self.thresholds:
            exp = self.setup_exp(threshold)
            acdcpp_heads, attrs = self.run_acdcpp(exp, threshold)

            if save_after_acdcpp:
                show(exp.corr, fname=f'ims/{self.run_name}/thresh{threshold}_before_acdc.png')
            
            acdc_heads, passes = self.run_acdc(exp)

            if save_after_acdc:
                show(exp.corr, fname=f'ims/{self.run_name}/thresh{threshold}_after_acdc.png')
            pruned_heads[threshold] = (acdcpp_heads, acdc_heads)
            num_passes[threshold] = passes
            pruned_attrs[threshold] = attrs
        return pruned_heads, num_passes, pruned_attrs