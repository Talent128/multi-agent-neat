import copy
import pickle
from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import TensorDictBase
from tensordict.utils import unravel_key_list
from torch import nn
from torchrl.data import Composite, Unbounded

from benchmarl.models.common import Model, ModelConfig

from .activations import normalize_output_groups
from .differentiable_recurrent_net import DifferentiableRecurrentNet


def _load_genome_package(package_path: str):
    with open(package_path, "rb") as f:
        package = pickle.load(f)
    if "genome" not in package or "neat_config" not in package:
        raise ValueError(
            f"Genome package at {package_path} must contain 'genome' and 'neat_config'."
        )
    return package


class BenchmarlRecurrentPolicy(Model):
    """BenchMARL-compatible actor wrapper around DifferentiableRecurrentNet."""

    def __init__(
        self,
        genome_package_path: str,
        prune_empty: bool,
        use_current_activs: bool,
        n_internal_steps: int,
        tanh_output_dist: bool,
        freeze_core: bool,
        std_bias_init: float,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        if not self.input_has_agent_dim or self.centralised:
            raise ValueError(
                "BenchmarlRecurrentPolicy currently only supports decentralized actor inputs with an agent dimension."
            )
        if not self.share_params:
            raise ValueError(
                "BenchmarlRecurrentPolicy expects shared actor parameters, matching the pretrained NEAT controller."
            )

        package = _load_genome_package(genome_package_path)
        self._source_package = package
        self._source_genome = copy.deepcopy(package["genome"])
        self._source_generation = package.get("generation")
        self._source_fitness = package.get("fitness")

        self.tanh_output_dist = tanh_output_dist
        self.std_bias_init = std_bias_init

        self.core = DifferentiableRecurrentNet.create(
            genome=package["genome"],
            config=package["neat_config"],
            batch_size=1,
            prune_empty=prune_empty,
            use_current_activs=use_current_activs,
            n_internal_steps=n_internal_steps,
            device=self.device,
        )

        if freeze_core:
            for param in self.core.parameters():
                param.requires_grad = False

        self.hidden_state_name = (self.agent_group, f"_hidden_neat_{self.model_index}")
        self.output_state_name = (self.agent_group, f"_prev_output_neat_{self.model_index}")
        self.rnn_keys = ["is_init", self.output_state_name]
        if self.core.n_hidden > 0:
            self.rnn_keys.append(self.hidden_state_name)
        self.rnn_keys = unravel_key_list(self.rnn_keys)
        self.in_keys += self.rnn_keys

        self.input_features = sum(spec.shape[-1] for spec in self.input_spec.values(True, True))
        self.output_features = self.output_leaf_spec.shape[-1]
        self.core_action_features = self.core.n_outputs

        if self.output_features not in (
            self.core_action_features,
            self.core_action_features * 2,
        ):
            raise ValueError(
                "BenchmarlRecurrentPolicy only supports actor output sizes equal to action_dim or 2 * action_dim."
            )

        if self.output_features == self.core_action_features * 2:
            self.scale_param = nn.Parameter(
                torch.full((self.core_action_features,), std_bias_init, device=self.device)
            )
        else:
            self.register_parameter("scale_param", None)

        action_space = self.action_spec[(self.agent_group, "action")].space
        action_low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float32)
        if action_low.ndim > 1:
            action_low = action_low[0]
        if action_high.ndim > 1:
            action_high = action_high[0]
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    def _perform_checks(self):
        super()._perform_checks()
        for input_key, input_spec in self.input_spec.items(True, True):
            if len(input_spec.shape) != 2:
                raise ValueError(
                    f"BenchmarlRecurrentPolicy expects per-agent vector inputs, got {input_key}: {input_spec.shape}"
                )
        if self.output_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "BenchmarlRecurrentPolicy expects the output spec to include the agent dimension."
            )

    def export_genome(self):
        genome = copy.deepcopy(self._source_genome)
        self.core.sync_to_genome(genome)
        return genome

    def export_package(self):
        package = copy.deepcopy(self._source_package)
        package["genome"] = self.export_genome()
        package["source_generation"] = self._source_generation
        package["source_fitness"] = self._source_fitness
        return package

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_tensor = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
            ],
            dim=-1,
        )
        prev_outputs = tensordict.get(self.output_state_name, None)
        prev_hidden = (
            tensordict.get(self.hidden_state_name, None)
            if self.core.n_hidden > 0
            else None
        )
        is_init = tensordict.get("is_init")

        output_tensor, next_hidden, next_outputs, missing_batch, training = self._rollout_core(
            input_tensor=input_tensor,
            is_init=is_init,
            prev_hidden=prev_hidden,
            prev_outputs=prev_outputs,
        )

        if missing_batch:
            output_tensor = output_tensor.squeeze(0)
            if next_hidden is not None:
                next_hidden = next_hidden.squeeze(0)
            next_outputs = next_outputs.squeeze(0)

        tensordict.set(self.out_key, output_tensor)
        if not training:
            if next_hidden is not None:
                tensordict.set(("next", *self.hidden_state_name), next_hidden)
            tensordict.set(("next", *self.output_state_name), next_outputs)
        return tensordict

    def _rollout_core(
        self,
        input_tensor: torch.Tensor,
        is_init: torch.Tensor,
        prev_hidden: Optional[torch.Tensor],
        prev_outputs: Optional[torch.Tensor],
    ):
        training = prev_outputs is None
        missing_batch = False

        if input_tensor.ndim == 2:
            missing_batch = True
            input_tensor = input_tensor.unsqueeze(0)
            is_init = is_init.unsqueeze(0)
            if prev_hidden is not None:
                prev_hidden = prev_hidden.unsqueeze(0)
            if prev_outputs is not None:
                prev_outputs = prev_outputs.unsqueeze(0)

        if not training:
            input_tensor = input_tensor.unsqueeze(1)
            if is_init.ndim == 2:
                is_init = is_init.unsqueeze(1)
        elif is_init.ndim == 2:
            is_init = is_init.unsqueeze(-1)

        if is_init.ndim == 2:
            is_init = is_init.unsqueeze(-1)

        batch, seq, n_agents, input_dim = input_tensor.shape
        if n_agents != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agents, got {n_agents}")
        if input_dim != self.input_features:
            raise ValueError(f"Expected input dim {self.input_features}, got {input_dim}")

        if prev_outputs is None:
            prev_outputs = torch.zeros(
                batch,
                self.n_agents,
                self.core.n_outputs,
                device=self.device,
                dtype=torch.float32,
            )
        if self.core.n_hidden > 0 and prev_hidden is None:
            prev_hidden = torch.zeros(
                batch,
                self.n_agents,
                self.core.n_hidden,
                device=self.device,
                dtype=torch.float32,
            )

        outputs = []
        hidden_state = prev_hidden
        output_state = prev_outputs

        for t in range(seq):
            reset_mask = is_init[:, t].to(torch.bool).view(batch, 1, 1)
            output_state = torch.where(reset_mask, torch.zeros_like(output_state), output_state)
            if hidden_state is not None:
                hidden_state = torch.where(reset_mask, torch.zeros_like(hidden_state), hidden_state)

            flat_input = input_tensor[:, t].reshape(batch * self.n_agents, self.input_features)
            flat_prev_output = output_state.reshape(batch * self.n_agents, self.core.n_outputs)
            flat_prev_hidden = (
                hidden_state.reshape(batch * self.n_agents, self.core.n_hidden)
                if hidden_state is not None
                else None
            )

            flat_next_hidden, flat_next_output = self._core_step(
                inputs=flat_input,
                prev_hidden=flat_prev_hidden,
                prev_outputs=flat_prev_output,
            )

            if hidden_state is not None:
                hidden_state = flat_next_hidden.view(batch, self.n_agents, self.core.n_hidden)
            output_state = flat_next_output.view(batch, self.n_agents, self.core.n_outputs)
            outputs.append(self._build_actor_output(output_state))

        output_tensor = torch.stack(outputs, dim=1)
        if not training:
            output_tensor = output_tensor[:, 0]
        return output_tensor, hidden_state, output_state, missing_batch, training

    def _core_step(
        self,
        inputs: torch.Tensor,
        prev_hidden: Optional[torch.Tensor],
        prev_outputs: torch.Tensor,
    ):
        activs_for_output = prev_hidden
        next_hidden = prev_hidden

        if self.core.n_hidden > 0:
            next_hidden = prev_hidden
            activs_for_output = prev_hidden
            for _ in range(self.core.n_internal_steps):
                hidden_pre = (
                    self.core.hidden_responses
                    * (
                        self.core.input_to_hidden.mm(inputs.t()).t()
                        + self.core.hidden_to_hidden.mm(next_hidden.t()).t()
                        + self.core.output_to_hidden.mm(prev_outputs.t()).t()
                    )
                    + self.core.hidden_biases
                )
                next_hidden = self.core.apply_hidden_activations(hidden_pre)
            if self.core.use_current_activs:
                activs_for_output = next_hidden

        output_inputs = (
            self.core.input_to_output.mm(inputs.t()).t()
            + self.core.output_to_output.mm(prev_outputs.t()).t()
        )
        if self.core.n_hidden > 0:
            output_inputs += self.core.hidden_to_output.mm(activs_for_output.t()).t()

        output_pre = self.core.output_responses * output_inputs + self.core.output_biases
        next_outputs = self.core.apply_output_activations(output_pre)
        return next_hidden, next_outputs

    def _build_actor_output(self, core_outputs: torch.Tensor) -> torch.Tensor:
        action_values = self._map_core_output_to_action(core_outputs)
        actor_param = self._action_to_actor_param(action_values)

        if self.scale_param is None:
            return actor_param

        scale = self.scale_param.view(*([1] * (actor_param.ndim - 1)), -1).expand_as(
            actor_param
        )
        return torch.cat([actor_param, scale], dim=-1)

    def _map_core_output_to_action(self, core_outputs: torch.Tensor) -> torch.Tensor:
        low = self.action_low.view(*([1] * (core_outputs.ndim - 1)), -1)
        high = self.action_high.view(*([1] * (core_outputs.ndim - 1)), -1)
        normalized = normalize_output_groups(core_outputs, self.core.output_activation_groups)
        return low + normalized * (high - low)

    def _action_to_actor_param(self, action_values: torch.Tensor) -> torch.Tensor:
        if not self.tanh_output_dist:
            return action_values

        low = self.action_low.view(*([1] * (action_values.ndim - 1)), -1)
        high = self.action_high.view(*([1] * (action_values.ndim - 1)), -1)
        center = (high + low) * 0.5
        scale = ((high - low) * 0.5).clamp_min(1e-6)
        normalized = ((action_values - center) / scale).clamp(-0.99999, 0.99999)
        return torch.atanh(normalized)


@dataclass
class BenchmarlRecurrentPolicyConfig(ModelConfig):
    genome_package_path: str
    prune_empty: bool = False
    use_current_activs: bool = False
    n_internal_steps: int = 1
    tanh_output_dist: bool = False
    freeze_core: bool = False
    std_bias_init: float = 0.0

    @staticmethod
    def associated_class():
        return BenchmarlRecurrentPolicy

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_prev_output_neat_{model_index}": Unbounded(
                    shape=(self._action_dim_from_package(),)
                )
            }
        )
        hidden_dim = self._hidden_dim_from_package()
        if hidden_dim > 0:
            spec.update(
                {
                    f"_hidden_neat_{model_index}": Unbounded(
                        shape=(hidden_dim,)
                    )
                }
            )
        return spec

    def _hidden_dim_from_package(self) -> int:
        package = _load_genome_package(self.genome_package_path)
        genome = package["genome"]
        config = package["neat_config"]
        return len(
            [k for k in genome.nodes.keys() if k not in config.genome_config.output_keys]
        )

    def _action_dim_from_package(self) -> int:
        package = _load_genome_package(self.genome_package_path)
        config = package["neat_config"]
        return len(config.genome_config.output_keys)
