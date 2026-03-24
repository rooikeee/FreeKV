import torch
from abc import ABC, abstractmethod
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)

class StepUpdater(ABC):
    def __init__(self, model: LlamaForCausalLM, tokenizer=None, sink=None, recent=None):
        self.model = model
        self.sink = sink
        self.recent = recent
        self.tokenizer = tokenizer

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def finish(self) -> dict:
        pass


class QuestUpdater(StepUpdater):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__(model)

    def reset(self, input: torch.Tensor):
        for layer in self.model.model.layers:
            layer.self_attn.num_pages = 0
            layer.self_attn._quest_token_select_events = []
            layer.self_attn._quest_attn_compute_events = []

    def update(self, pred_token_idx: int):
        pass

    def finish(self):
        if torch.cuda.is_available():
            synced_devices = set()
            for layer in self.model.model.layers:
                dev = layer.self_attn.q_proj.weight.device
                if dev.type != "cuda":
                    continue
                dev_idx = int(dev.index) if dev.index is not None else 0
                if dev_idx in synced_devices:
                    continue
                torch.cuda.synchronize(dev)
                synced_devices.add(dev_idx)

        layer_token_select_ms = []
        layer_attn_compute_ms = []
        layer_token_select_calls = []
        layer_attn_compute_calls = []

        for layer in self.model.model.layers:
            attn = layer.self_attn
            select_events = getattr(attn, "_quest_token_select_events", [])
            attn_events = getattr(attn, "_quest_attn_compute_events", [])
            sel_ms = 0.0
            attn_ms = 0.0
            for start_evt, end_evt in select_events:
                sel_ms += float(start_evt.elapsed_time(end_evt))
            for start_evt, end_evt in attn_events:
                attn_ms += float(start_evt.elapsed_time(end_evt))
            layer_token_select_ms.append(sel_ms)
            layer_attn_compute_ms.append(attn_ms)
            layer_token_select_calls.append(len(select_events))
            layer_attn_compute_calls.append(len(attn_events))

        return {
            "token_select_ms": float(sum(layer_token_select_ms)),
            "attn_compute_ms": float(sum(layer_attn_compute_ms)),
            "token_select_calls": int(sum(layer_token_select_calls)),
            "attn_compute_calls": int(sum(layer_attn_compute_calls)),
            "layer_token_select_ms": layer_token_select_ms,
            "layer_attn_compute_ms": layer_attn_compute_ms,
            "layer_token_select_calls": layer_token_select_calls,
            "layer_attn_compute_calls": layer_attn_compute_calls,
        }


class SpecRetUpdater(QuestUpdater):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__(model)

    def reset(self, input: torch.Tensor):
        super().reset(input)
        for layer in self.model.model.layers:
            layer.self_attn.num_correct = 0
            layer.self_attn.num_correct_kv_heads = torch.tensor([0], device=self.model.device)

    def finish(self):
        stats = super().finish()
        num_correct_layers = []
        num_correct_kv_heads_layers = []
        for layer in self.model.model.layers:
            num_correct_layers.append(layer.self_attn.num_correct)
            num_correct_kv_heads_layers.append(layer.self_attn.num_correct_kv_heads.item())
        stats.update({
            "num_correct": num_correct_layers,
            "num_correct_kv_heads": num_correct_kv_heads_layers,
        })
        return stats


class RaaSUpdater(StepUpdater):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__(model)

    def reset(self, input: torch.Tensor):
        for layer in self.model.model.layers:
            if layer.self_attn.token_budget > 0:
                layer.self_attn.num_pages = 0
                layer.self_attn.page_timestamp.fill_(0)

    def update(self, pred_token_idx: int):
        pass

    def finish(self):
        return {}
