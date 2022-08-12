import re
from typing import NamedTuple
import jax.numpy as jnp
from tqdm import tqdm
from jax.lax import cond, while_loop
from jax.experimental import host_callback
from jaxns.internals.types import NestedSamplerState

def _iterate(func):
    def wrapper(arg):
        i, body_state = arg
        return i + 1, func(body_state)
    return wrapper

def while_collect(cond_fun, body_fun, init_val, show_progress_bar=True):
    # A while loop where the id is tracked
    new_cond_fun = lambda arg: cond_fun(arg[1])
    new_body_fun = _iterate(body_fun)

    if show_progress_bar:
        progress_bar = ProgressBar()
        new_body_fun = progress_bar(new_body_fun)
    
    _, result = while_loop(
        new_cond_fun,
        new_body_fun,
        (0, init_val),
    )
    
    return result


class ProgressBar:
    """TODO: maybe different postfix depending whether dynamic or static"""
    def __init__(self):
        self.pbar = None

    def init_pbar(self, _arg, _transform, device):
        self.pbar = tqdm(position=device.id)
        self.pbar.set_description_str(f"Running on {device}", refresh=False)
        # TODO: add extra info from init state
        
    def update_pbar(self, arg, _transform, device):
        step, log_Z, num_likelihood_evals = arg
        self.pbar.update()
        self.pbar.set_postfix(step=step, log_Z=log_Z, num_likelihood_evals=num_likelihood_evals)

    def close_pbar(self, _arg, _transform, device):
        self.pbar.close()
        self.pbar = None

    def update(self, _i, new_state):
        """Update progress bar for current state.
        
        Host callback is required to modify attributes of ProgressBar.
        """
        step = new_state.step_idx
        log_Z = new_state.evidence_calculation.log_Z_mean
        num_likelihood_evals = new_state.num_likelihood_evaluations
        
        # Update
        host_callback.id_tap(
            self.update_pbar, (step, log_Z, num_likelihood_evals), tap_with_device=True
        )

        # Update TODO: add print_rate stuff and make use of i
        # _ = cond(
        #     remainder == 0,
        #     lambda _: host_callback.id_tap(
        #         self.update_pbar, (self.print_rate, state.step_idx, log_ZX), result=self.iter_num
        #     ),
        #     lambda _: self.iter_num,
        #     operand=None,
        # )
        
        # Close
        # TODO: add print_rate remainder on close
        _ = cond(
            new_state.done,
            lambda _: host_callback.id_tap(
                self.close_pbar, (), result=True, tap_with_device=True
            ),
            lambda _: True,
            operand=None,
        )

    def decorator(self, func):
        def wrapper(arg):
            i, _ = arg
            
            # Initialize pbar
            _ = cond(
                i == 0,
                lambda _: host_callback.id_tap(self.init_pbar, (), tap_with_device=True, result=True),
                lambda _: True,
                operand=None
            )
            
            i, result = func(arg)  # This should increment i by 1

            # TODO: a neater way of doing this
            if isinstance(result, NestedSamplerState):
                self.update(i, result)
            elif isinstance(result[0], NestedSamplerState):
                self.update(i, result[0])
            else:
                raise NotImplementedError("ProgressBar only works with NestedSamplerState")
            return i, result

        return wrapper

    def __call__(self, func):
        return self.decorator(func)
