"""Unified batched probing wrappers for Search-R1 and R1-Searcher."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import torch

from pyterrier_rag.frameworks.agentic import R1Searcher, SearchR1

from .backend import ProbedHuggingFaceBackend
from .confidence import mean_token_logprob


class _BatchedAnswerProbingMixin:
    """Shared batched intermediate-answer probing logic for agentic RAG models.

    Each active trajectory is probed once after its current reasoning response.
    The probe forces an ``<answer>`` continuation, is decoded deterministically,
    and records the answer text, raw token log-probabilities, and the
    length-normalized scalar confidence used by the paper.
    """

    def _build_probe_prompt(self, context: str, output: str) -> tuple[str, bool]:
        raise NotImplementedError

    def _probe_batch(self, prompts: List[str]) -> List[tuple[Optional[str], Any]]:
        generations = self.backend.generate(
            prompts,
            stop_sequences=[self.end_answer_tag],
            return_logprobs=True,
            deterministic=True,
        )

        results = []
        for generation in generations:
            text = getattr(generation, "text", None)
            answer = None
            if text:
                if self.start_answer_tag in text:
                    tail = text.split(self.start_answer_tag, 1)[1]
                else:
                    tail = text
                answer = tail.split(self.end_answer_tag, 1)[0].strip()
            results.append((answer, getattr(generation, "logprobs", None)))
        return results

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(df, ["query"])
        result_columns = [
            "qid",
            "query",
            "context",
            "search_history",
            "search_iterations",
            "qanswer",
            "output",
            "stop_reason",
            "probed_answers_after_think",
            "probed_logits_after_think",
            "probed_probs_after_think",
            "has_thought_record",
        ]
        if len(df) == 0:
            return pd.DataFrame([], columns=result_columns)

        active: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            active.append(
                {
                    "qid": str(row["qid"]),
                    "query": row["query"],
                    "context": self.get_prompt(row["query"]),
                    "search_history": [],
                    "search_iterations": 0,
                    "qanswer": "",
                    "output": "",
                    "stop_reason": None,
                    "probed_answers_after_think": [],
                    "probed_logits_after_think": [],
                    "probed_probs_after_think": [],
                    "has_thought_record": [],
                }
            )

        finished: List[Dict[str, Any]] = []

        for _ in range(self.max_turn):
            if not active:
                break

            outputs = self.generate([state["context"] for state in active])
            answers = self.check_answers(outputs)

            probe_prompts = []
            probe_has_thought = []
            for state, output in zip(active, outputs):
                prompt, has_thought = self._build_probe_prompt(state["context"], output)
                probe_prompts.append(prompt)
                probe_has_thought.append(has_thought)

            probe_results = self._probe_batch(probe_prompts)
            pending: List[Dict[str, Any]] = []

            for state, output, answer, has_thought, probe_result in zip(
                active,
                outputs,
                answers,
                probe_has_thought,
                probe_results,
            ):
                probed_answer, probed_logprobs = probe_result
                state["probed_answers_after_think"].append(probed_answer)
                state["probed_logits_after_think"].append(probed_logprobs)
                state["probed_probs_after_think"].append(
                    mean_token_logprob(probed_logprobs)
                )
                state["has_thought_record"].append(has_thought)

                state["context"] += "\n\n" + output
                state["output"] += "\n\n" + output

                if answer is not None:
                    state["qanswer"] = answer
                    state["stop_reason"] = "Got answer"
                    if self.end_answer_tag and not output.endswith(self.end_answer_tag):
                        state["context"] += self.end_answer_tag
                        state["output"] += self.end_answer_tag
                    finished.append(state)
                    continue

                next_search = self.extract_search_query(output)
                if not next_search:
                    state["stop_reason"] = "No answer, no search"
                    finished.append(state)
                    continue

                if self.end_search_tag and not output.endswith(self.end_search_tag):
                    state["context"] += self.end_search_tag
                    state["output"] += self.end_search_tag

                state["search_history"].append(next_search)
                state["search_iterations"] += 1
                pending.append(state)

            if not pending:
                active = []
                break

            batch_queries = pd.DataFrame(
                {
                    "qid": [state["qid"] for state in pending],
                    "query": [state["search_history"][-1] for state in pending],
                }
            )
            batch_results = (self.retriever % self.top_k).transform(batch_queries)
            qid_to_docs = self.batch_format_docs(
                self._restructure_search_results(batch_results),
                states=pending,
            )

            next_active = []
            for state in pending:
                docs = qid_to_docs.get(state["qid"])
                if docs:
                    state["context"] += docs
                    next_active.append(state)
                else:
                    state["stop_reason"] = "No retrieval results"
                    finished.append(state)
            active = next_active

        for state in active:
            if not state.get("stop_reason"):
                state["stop_reason"] = "No answer after max turns"

        return pd.DataFrame(finished + active, columns=result_columns)


class ProbedSearchR1(_BatchedAnswerProbingMixin, SearchR1):
    """Batched Search-R1 with after-reasoning intermediate-answer probing."""

    @classmethod
    def from_hf(
        cls,
        retriever: pt.Transformer,
        model: str = SearchR1.DEFAULT_MODEL,
        backend_args: dict | None = None,
        **kwargs,
    ) -> "ProbedSearchR1":
        if backend_args is None:
            backend_args = {
                "model_args": {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                },
                "generation_args": {
                    "temperature": 0.7,
                    "max_new_tokens": 1024,
                    "do_sample": True,
                },
            }

        backend = ProbedHuggingFaceBackend(model, **backend_args)
        backend.tokenizer.padding_side = "left"
        backend._model.generation_config.pad_token_id = backend.tokenizer.pad_token_id
        return cls(retriever, backend=backend, **kwargs)

    def _build_probe_prompt(self, context: str, output: str) -> tuple[str, bool]:
        # Search-R1 emits complete <think>...</think> spans before either
        # searching or answering. Keep only the current reasoning span and
        # force an answer continuation for the probe.
        if self.start_answer_tag in output:
            prefix = output.split(self.start_answer_tag, 1)[0]
            return context + "\n\n" + prefix + self.start_answer_tag, True

        think_end = output.find("</think>")
        if think_end >= 0:
            reasoning_prefix = output[: think_end + len("</think>")]
            return context + "\n\n" + reasoning_prefix + "\n\n" + self.start_answer_tag, True

        return context + "\n\n" + self.start_answer_tag, False


class ProbedR1Searcher(_BatchedAnswerProbingMixin, R1Searcher):
    """Batched R1-Searcher with after-reasoning intermediate-answer probing."""

    @classmethod
    def from_hf(
        cls,
        retriever: pt.Transformer,
        model: str = R1Searcher.DEFAULT_MODEL,
        backend_args: dict | None = None,
        **kwargs,
    ) -> "ProbedR1Searcher":
        if backend_args is None:
            backend_args = {
                "model_args": {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "trust_remote_code": True,
                },
                "generation_args": {
                    "max_new_tokens": 512,
                    "do_sample": False,
                },
            }

        backend = ProbedHuggingFaceBackend(model, **backend_args)
        backend.tokenizer.padding_side = "left"
        backend._model.generation_config.pad_token_id = backend.tokenizer.pad_token_id
        return cls(retriever, backend=backend, **kwargs)

    def _build_probe_prompt(self, context: str, output: str) -> tuple[str, bool]:
        # R1-Searcher prompts begin inside <think>. When generation stops at a
        # search boundary the current reasoning therefore needs an explicit
        # </think> before the forced answer.
        if self.start_answer_tag in output:
            prefix = output.split(self.start_answer_tag, 1)[0]
            return context + "\n\n" + prefix + self.start_answer_tag, True

        if self.start_search_tag in output:
            reasoning_prefix = output.split(self.start_search_tag, 1)[0]
        else:
            reasoning_prefix = output

        return (
            context
            + "\n\n"
            + reasoning_prefix.rstrip()
            + "</think>\n\n"
            + self.start_answer_tag,
            bool(reasoning_prefix.strip()),
        )


__all__ = ["ProbedSearchR1", "ProbedR1Searcher"]
