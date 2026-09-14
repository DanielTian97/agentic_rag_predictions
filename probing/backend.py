"""Hugging Face backend helpers for deterministic answer probing."""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import StoppingCriteriaList

from pyterrier_rag.backend import BackendOutput, HuggingFaceBackend, StopWordCriteria


class ProbedHuggingFaceBackend(HuggingFaceBackend):
    """Hugging Face backend that can return per-token generation log-probabilities.

    Normal agent generation behaves exactly like :class:`HuggingFaceBackend`.
    Probe generation can additionally request token log-probabilities and disable
    sampling so that intermediate answers are generated deterministically.
    """

    supports_logprobs = True

    @torch.no_grad()
    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        num_responses: int = 1,
        deterministic: bool = False,
    ) -> List[BackendOutput]:
        if not isinstance(inps, list):
            raise TypeError(f"Expected list input, found {type(inps)!r}")
        if not inps:
            return []
        if not isinstance(inps[0], str):
            raise ValueError(f"{self!r} only supports string inputs")
        if num_responses != 1:
            raise ValueError(f"{self!r} does not support num_responses > 1")

        inputs = self.tokenizer(
            inps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        model_device = self._model.get_input_embeddings().weight.device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_args = dict(self._generation_args)
        if max_new_tokens is not None:
            generation_args["max_new_tokens"] = max_new_tokens
        if deterministic:
            generation_args["do_sample"] = False
            generation_args.pop("temperature", None)
            generation_args.pop("top_p", None)
            generation_args.pop("top_k", None)

        stop_criteria = None
        if stop_sequences is not None:
            stop_criteria = StopWordCriteria(
                tokenizer=self.tokenizer,
                prompt_size=inputs["input_ids"].shape[-1],
                stop_words=stop_sequences,
                check_every=1,
            )
            generation_args["stopping_criteria"] = StoppingCriteriaList([stop_criteria])

        outputs = self._model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
            **generation_args,
        )

        input_ids = inputs["input_ids"]
        prompt_width = input_ids.shape[1]
        sequences = outputs["sequences"]

        if self._remove_prompt:
            if self.tokenizer.padding_side == "left":
                generated_sequences = [seq[prompt_width:] for seq in sequences]
            else:
                pad_token_id = self.tokenizer.pad_token_id
                prompt_lengths = (input_ids != pad_token_id).sum(dim=1).tolist()
                generated_sequences = [
                    sequences[i, prompt_length:]
                    for i, prompt_length in enumerate(prompt_lengths)
                ]
        else:
            generated_sequences = [seq for seq in sequences]

        texts = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        if stop_criteria is not None:
            if not self._remove_prompt:
                raise NotImplementedError(
                    "Removing stop sequences requires prompt removal in backend outputs"
                )
            for idx, text in enumerate(texts):
                for sequence in stop_sequences or []:
                    text = text.rsplit(sequence, maxsplit=1)[0]
                texts[idx] = text

        if not return_logprobs:
            return [BackendOutput(text=text) for text in texts]

        scores = outputs["scores"]
        backend_outputs: List[BackendOutput] = []
        for batch_idx, generated in enumerate(generated_sequences):
            generated = torch.as_tensor(generated, device=model_device)
            step_count = min(len(scores), generated.shape[0])
            token_logprobs = []
            for step_idx in range(step_count):
                step_log_probs = F.log_softmax(scores[step_idx][batch_idx], dim=-1)
                token_logprobs.append(step_log_probs[generated[step_idx]].item())
            backend_outputs.append(
                BackendOutput(text=texts[batch_idx], logprobs=token_logprobs)
            )

        return backend_outputs
