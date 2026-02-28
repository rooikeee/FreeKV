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

    def update(self, pred_token_idx: int):
        pass

    def finish(self):
        return {}


class SpecRetUpdater(QuestUpdater):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__(model)

    def reset(self, input: torch.Tensor):
        for layer in self.model.model.layers:
            layer.self_attn.num_pages = 0
            layer.self_attn.num_correct = 0
            layer.self_attn.num_correct_kv_heads = torch.tensor([0], device=self.model.device)

    def finish(self):
        num_correct_layers = []
        num_correct_kv_heads_layers = []
        for layer in self.model.model.layers:
            num_correct_layers.append(layer.self_attn.num_correct)
            num_correct_kv_heads_layers.append(layer.self_attn.num_correct_kv_heads.item())
        return {"num_correct": num_correct_layers, "num_correct_kv_heads": num_correct_kv_heads_layers}


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
