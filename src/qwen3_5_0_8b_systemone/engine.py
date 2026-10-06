from __future__ import annotations

import copy
import io
import itertools
import json
import os
import string
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from PIL import Image, ImageOps

from .contract import (
    Attachment,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from .scoring import answer_from_logits

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_ALIAS = "qwen3.5-0.8b-systemone"
ANSWER_PREFIX = "\nAnswer:"
MAX_CONTEXT_TOKENS = 8_192
MAX_SUFFIX_TOKENS = 65_536
MAX_DIRECT_INPUT_TOKENS = 128
MAX_DIRECT_SINGLE_INPUT_TOKENS = 2_048
GRAPH_WIDTH_STEP = 16


class EngineBusyError(RuntimeError):
    pass


class ContextLimitError(ValueError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompiledQuestion:
    question_id: str
    question: Question
    input_ids: tuple[int, ...]
    candidate_token_ids: list[int]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _safe(value: object) -> object:
    if isinstance(value, str):
        return value.replace("<|", "<\u200b|")
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            key.replace("<|", "<\u200b|"): _safe(item) for key, item in value.items()
        }
    return value


class SystemOneEngine:
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: Any = None
        self.labels: list[tuple[str, int]] = []
        self._inference_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.batch_size = int(os.getenv("SYSTEMONE_BATCH_SIZE", "16"))
        self.cuda_graphs = os.getenv("SYSTEMONE_CUDA_GRAPHS", "1") != "0"
        self._graphs: dict[tuple[str, int, int], tuple[Any, dict[str, Any], Any]] = {}
        self._graph_pool: Any = None
        print(f"Qwen3.5 System One batch size: {self.batch_size}")
        print(f"Qwen3.5 System One CUDA graphs: {self.cuda_graphs}")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            if not torch.cuda.is_available():
                raise RuntimeError("Qwen3.5 System One requires a CUDA GPU")
            self.device = torch.device("cuda")
            revision = os.getenv("SYSTEMONE_MODEL_REVISION", MODEL_REVISION)
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
            image_min_pixels = os.getenv("SYSTEMONE_IMAGE_MIN_PIXELS")
            if image_min_pixels:
                self.processor.image_processor.size.shortest_edge = int(
                    image_min_pixels
                )
                print(f"Qwen3.5 System One image minimum: {image_min_pixels} pixels")
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                revision=revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self._linearize_patch_embed()
            self.labels = self._verified_labels()

    def _linearize_patch_embed(self) -> None:
        """Run the vision patch embedding as a matmul.

        Its Conv3d kernel equals its stride and covers each whole patch, so every
        output is a dot product over one flattened patch. cuDNN picks a
        pathologically slow bf16 algorithm for this shape here: a 448x112 frame
        spent 7.6 s in the convolution alone.
        """
        import torch.nn.functional as F

        patch_embed = self.model.model.visual.patch_embed
        weight = patch_embed.proj.weight.reshape(patch_embed.embed_dim, -1)
        bias = patch_embed.proj.bias

        def forward(hidden_states: Any) -> Any:
            flat = hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype)
            return F.linear(flat, weight, bias)

        patch_embed.forward = forward

    def _verified_labels(self) -> list[tuple[str, int]]:
        tokenizer = self.processor.tokenizer
        prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
        labels: list[tuple[str, int]] = []
        seen: set[int] = set()
        candidates = (
            "".join(parts)
            for size in (1, 2)
            for parts in itertools.product(string.ascii_uppercase, repeat=size)
        )
        for label in candidates:
            full = tokenizer.encode(
                f"{ANSWER_PREFIX} {label}", add_special_tokens=False
            )
            if (
                len(full) == len(prefix_ids) + 1
                and full[: len(prefix_ids)] == prefix_ids
                and full[-1] not in seen
            ):
                labels.append((label, full[-1]))
                seen.add(full[-1])
            if len(labels) == 255:
                return labels
        raise RuntimeError(
            f"tokenizer provides only {len(labels)} verified candidate labels; 255 are required"
        )

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if not self._inference_lock.acquire(blocking=False):
            raise EngineBusyError("the inference engine is busy")
        try:
            self.load()
            return self._evaluate(request)
        finally:
            if self.model is not None:
                self.model.model.rope_deltas = None
            self._inference_lock.release()

    def _evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        import torch

        images = self._decode_images(request.attachments)
        try:
            shared_inputs = self._shared_inputs(request, images)
            shared_inputs = {
                key: value.to(self.device) for key, value in shared_inputs.items()
            }
            prefix_length = int(shared_inputs["input_ids"].shape[1])
            compiled = [
                self._compile(question_id, question)
                for question_id, question in request.questions.items()
            ]
            suffix_tokens = sum(len(item.input_ids) for item in compiled)
            if suffix_tokens > MAX_SUFFIX_TOKENS:
                raise ContextLimitError(
                    f"question suffixes exceed {MAX_SUFFIX_TOKENS} tokens"
                )
            longest = max(len(item.input_ids) for item in compiled)
            if prefix_length + longest > MAX_CONTEXT_TOKENS:
                raise ContextLimitError(
                    f"shared state plus a question exceeds {MAX_CONTEXT_TOKENS} tokens"
                )
        except Exception:
            for image in images:
                image.close()
            raise

        try:
            answers: dict[str, Any] = {}
            batch_size = max(1, self.batch_size)
            direct_single = (
                len(compiled) == 1
                and prefix_length + longest <= MAX_DIRECT_SINGLE_INPUT_TOKENS
            )
            direct_short_text = (
                not images
                and len(compiled) <= batch_size
                and prefix_length + longest <= MAX_DIRECT_INPUT_TOKENS
            )
            if direct_single or direct_short_text:
                logits = self._evaluate_direct_batch(
                    compiled,
                    shared_inputs,
                )
                for item, values in zip(compiled, logits, strict=True):
                    answers[item.question_id] = answer_from_logits(
                        item.question, values
                    )
                return SystemOneResponse(
                    answers=answers,
                    usage=Usage(
                        input_tokens=prefix_length + suffix_tokens, output_tokens=0
                    ),
                )

            with torch.inference_mode():
                prefix = self.model.model(
                    **shared_inputs, use_cache=True, return_dict=True
                )
            base_cache = prefix.past_key_values
            rope_deltas = (
                None
                if self.model.model.rope_deltas is None
                else self.model.model.rope_deltas.clone()
            )
            offset = 0
            while offset < len(compiled):
                batch = compiled[offset : offset + batch_size]
                try:
                    logits = self._evaluate_batch(
                        batch, base_cache, shared_inputs["attention_mask"], rope_deltas
                    )
                except torch.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise InferenceError(
                            "GPU memory exhausted while evaluating one question"
                        ) from error
                    batch_size = max(1, batch_size // 2)
                    continue
                for item, values in zip(batch, logits, strict=True):
                    answers[item.question_id] = answer_from_logits(
                        item.question, values
                    )
                offset += len(batch)
            return SystemOneResponse(
                answers=answers,
                usage=Usage(
                    input_tokens=prefix_length + suffix_tokens, output_tokens=0
                ),
            )
        finally:
            for image in images:
                image.close()
            self.model.model.rope_deltas = None

    def _shared_inputs(
        self, request: SystemOneRequest, images: list[Image.Image]
    ) -> dict[str, Any]:
        system = (
            "Select one listed label for each question using the shared state. "
            "Answer with that label only."
        )
        state_text = f"State:\n{_json(_safe(request.state))}"
        if images:
            image_names = ", ".join(attachment.id for attachment in request.attachments)
            content: Any = [*(dict(type="image", image=image) for image in images)]
            content.append(
                {
                    "type": "text",
                    "text": f"Attachments in order: {image_names}\n{state_text}",
                }
            )
        else:
            content = state_text
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

    def _compile(self, question_id: str, question: Question) -> CompiledQuestion:
        if isinstance(question, ChoiceQuestion):
            criteria = [
                [_safe(name), _safe(description)]
                for name, description in question.criteria.items()
            ]
        elif isinstance(question, ScoreQuestion):
            criteria = [
                [index, _safe(description)]
                for index, description in enumerate(question.criteria)
            ]
        elif isinstance(question, NoulQuestion):
            descriptions = question.criteria or {}
            true_description = descriptions.true if descriptions else None
            false_description = descriptions.false if descriptions else None
            criteria = [
                [
                    True,
                    _safe("yes" if true_description is None else true_description),
                ],
                [
                    False,
                    _safe("no" if false_description is None else false_description),
                ],
            ]
        else:
            raise TypeError(f"unsupported question type: {type(question).__name__}")

        prompt = "\n".join(
            [f"Question={_json(_safe(question.instructions))}"]
            + [
                f"{self.labels[index][0]}={_json(criterion)}"
                for index, criterion in enumerate(criteria)
            ]
        )
        input_ids = self._branch_input_ids(prompt)
        return CompiledQuestion(
            question_id=question_id,
            question=question,
            input_ids=input_ids,
            candidate_token_ids=[
                token_id for _, token_id in self.labels[: len(criteria)]
            ],
        )

    @lru_cache(maxsize=256)
    def _branch_input_ids(self, prompt: str) -> tuple[int, ...]:
        branch = (
            self.processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ANSWER_PREFIX
        )
        return tuple(
            self.processor.tokenizer.encode(branch, add_special_tokens=False)
        )

    @staticmethod
    def _fork_cache(base_cache: Any, count: int, device: Any) -> Any:
        """Copy cache metadata and expand each tensor directly from the shared row."""
        import torch

        cache = copy.copy(base_cache)
        cache.layers = [copy.copy(layer) for layer in base_cache.layers]
        for layer in cache.layers:
            for name, value in vars(layer).items():
                if isinstance(value, dict):
                    setattr(layer, name, value.copy())
        cache.reorder_cache(torch.zeros(count, dtype=torch.long, device=device))
        return cache

    def _evaluate_direct_batch(
        self,
        batch: list[CompiledQuestion],
        shared_inputs: dict[str, Any],
    ) -> list[list[float]]:
        import torch

        if self.cuda_graphs and "pixel_values" not in shared_inputs:
            prefix = tuple(shared_inputs["input_ids"][0].tolist())
            final_hidden = self._graph_hidden([prefix + item.input_ids for item in batch])
            if final_hidden is not None:
                return self._project_candidates(batch, final_hidden)

        prefix_ids = shared_inputs["input_ids"]
        prefix_mask = shared_inputs["attention_mask"]
        count = len(batch)
        prefix_length = int(prefix_ids.shape[1])
        lengths = torch.tensor(
            [prefix_length + len(item.input_ids) for item in batch],
            device=self.device,
        )
        width = int(lengths.max().item())
        input_ids = torch.full(
            (count, width),
            self.processor.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, :prefix_length] = prefix_ids.repeat(count, 1)
        attention_mask[:, :prefix_length] = prefix_mask.repeat(count, 1)
        for row, item in enumerate(batch):
            end = prefix_length + len(item.input_ids)
            input_ids[row, prefix_length:end] = torch.tensor(
                item.input_ids, device=self.device
            )
            attention_mask[row, prefix_length:end] = 1

        model_inputs = {
            key: value
            for key, value in shared_inputs.items()
            if key
            not in {
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "mm_token_type_ids",
            }
        }
        for name in ("token_type_ids", "mm_token_type_ids"):
            if name not in shared_inputs:
                continue
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids[:, :prefix_length] = shared_inputs[name].repeat(count, 1)
            model_inputs[name] = token_type_ids

        if self.cuda_graphs and count == 1:
            final_hidden = self._graph_hidden_multimodal(
                input_ids, attention_mask, model_inputs
            )
            if final_hidden is not None:
                return self._project_candidates(batch, final_hidden)

        with torch.inference_mode():
            output = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_inputs,
                use_cache=False,
                return_dict=True,
            )
        final_hidden = output.last_hidden_state[
            torch.arange(count, device=self.device), lengths - 1
        ].float()
        return self._project_candidates(batch, final_hidden)

    def _graph_hidden(self, rows: list[tuple[int, ...]]) -> Any:
        """Return each row's final hidden state from a captured text-only forward.

        Rows are right-padded to a width bucket. Every Qwen3.5 layer is causal, so
        padding after a row's last token cannot change that token, and the forward
        needs no attention mask. Text-only rope positions are a plain arange. Graphs
        are keyed by exact batch size because padded rows cost real GPU time once
        the forward is compute-bound.
        """
        import torch

        batch = len(rows)
        width = -(-max(map(len, rows)) // GRAPH_WIDTH_STEP) * GRAPH_WIDTH_STEP
        entry = self._graph_entry(
            ("text", batch, width),
            lambda: {
                "input_ids": torch.full(
                    (batch, width),
                    self.processor.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
            },
        )
        if entry is None:
            return None
        graph, inputs, hidden = entry

        host = torch.full(
            (batch, width), self.processor.tokenizer.pad_token_id, dtype=torch.long
        )
        for index, row in enumerate(rows):
            host[index, : len(row)] = torch.tensor(row)
        inputs["input_ids"].copy_(host)
        graph.replay()
        last = torch.tensor([len(row) - 1 for row in rows], device=self.device)
        return hidden[torch.arange(len(rows), device=self.device), last].float()

    def _graph_hidden_multimodal(
        self, input_ids: Any, attention_mask: Any, model_inputs: dict[str, Any]
    ) -> Any:
        """Return the final hidden state of one image row from a captured forward.

        The vision encoder and multimodal rope positions run eagerly once per
        request. Only the language model, which dominates, replays as a graph
        over the merged embeddings, right-padded to a width bucket.
        """
        import torch

        model = self.model.model
        length = int(input_ids.shape[1])
        width = -(-length // GRAPH_WIDTH_STEP) * GRAPH_WIDTH_STEP
        hidden_size = model.config.text_config.hidden_size
        dtype = model.language_model.embed_tokens.weight.dtype
        entry = self._graph_entry(
            ("image", 1, width),
            lambda: {
                "inputs_embeds": torch.zeros(
                    (1, width, hidden_size), dtype=dtype, device=self.device
                ),
                "position_ids": torch.zeros(
                    (3, 1, width), dtype=torch.long, device=self.device
                ),
            },
        )
        if entry is None:
            return None
        graph, inputs, hidden = entry

        with torch.inference_mode():
            embeds = model.get_input_embeddings()(input_ids)
            if "pixel_values" in model_inputs:
                image_embeds = model.get_image_features(
                    model_inputs["pixel_values"],
                    model_inputs["image_grid_thw"],
                    return_dict=True,
                ).pooler_output
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
                embeds = embeds.masked_scatter(
                    image_mask, torch.cat(image_embeds).to(embeds.dtype)
                )
            positions = model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=embeds,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
                mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
            )
            if positions is None:
                return None
            inputs["inputs_embeds"].zero_()
            inputs["inputs_embeds"][:, :length] = embeds
            inputs["position_ids"].zero_()
            inputs["position_ids"][:, :, :length] = positions
        graph.replay()
        return hidden[:, length - 1].float()

    def _graph_entry(
        self, key: tuple[str, int, int], make_inputs: Any
    ) -> tuple[Any, dict[str, Any], Any] | None:
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture_graph(make_inputs())
            except Exception as error:
                print(f"Qwen3.5 System One CUDA graphs disabled: {error!r}")
                self.cuda_graphs = False
                self._graphs.clear()
                return None
            self._graphs[key] = entry
        return entry

    def _capture_graph(
        self, inputs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any], Any]:
        import torch

        language_model = self.model.model.language_model
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        # Warm up on a side stream so cuBLAS workspaces and Triton autotuning
        # finish before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                language_model(**inputs, use_cache=False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._graph_pool):
            hidden = language_model(**inputs, use_cache=False).last_hidden_state
        return graph, inputs, hidden

    def _project_candidates(
        self, batch: list[CompiledQuestion], final_hidden: Any
    ) -> list[list[float]]:
        import torch

        weight = self.model.lm_head.weight
        logits = [
            torch.mv(
                weight.index_select(
                    0, torch.tensor(item.candidate_token_ids, device=self.device)
                ).float(),
                final_hidden[row],
            )
            for row, item in enumerate(batch)
        ]
        values = torch.cat(logits).cpu().tolist()
        results: list[list[float]] = []
        offset = 0
        for item in batch:
            size = len(item.candidate_token_ids)
            results.append(values[offset : offset + size])
            offset += size
        return results

    def _evaluate_batch(
        self,
        batch: list[CompiledQuestion],
        base_cache: Any,
        prefix_mask: Any,
        rope_deltas: Any,
    ) -> list[list[float]]:
        import torch

        count = len(batch)
        lengths = torch.tensor(
            [len(item.input_ids) for item in batch], device=self.device
        )
        width = int(lengths.max().item())
        pad_id = self.processor.tokenizer.pad_token_id
        suffix_ids = torch.full(
            (count, width), pad_id, dtype=torch.long, device=self.device
        )
        suffix_mask = torch.zeros((count, width), dtype=torch.long, device=self.device)
        for row, item in enumerate(batch):
            size = len(item.input_ids)
            suffix_ids[row, :size] = torch.tensor(item.input_ids, device=self.device)
            suffix_mask[row, :size] = 1

        full_mask = torch.cat([prefix_mask.repeat(count, 1), suffix_mask], dim=1)
        positions = full_mask.cumsum(-1)[:, -width:] - 1
        positions.masked_fill_(suffix_mask == 0, 0)
        if rope_deltas is not None:
            delta = rope_deltas.repeat_interleave(count, dim=0).view(1, count, 1)
            positions = positions.unsqueeze(0).expand(3, -1, -1) + delta

        cache = self._fork_cache(base_cache, count, self.device)
        self.model.model.rope_deltas = rope_deltas
        with torch.inference_mode():
            output = self.model.model(
                input_ids=suffix_ids,
                attention_mask=full_mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        final_hidden = output.last_hidden_state[
            torch.arange(count, device=self.device), lengths - 1
        ].float()
        return self._project_candidates(batch, final_hidden)

    @staticmethod
    def _decode_image(attachment: Attachment) -> Image.Image:
        image = Image.open(io.BytesIO(attachment.decoded()))
        expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[
            attachment.media_type
        ]
        if image.format != expected:
            image.close()
            raise ValueError(
                f"attachment {attachment.id!r} does not match {attachment.media_type}"
            )
        if image.width * image.height > 20_000_000:
            image.close()
            raise ValueError(f"attachment {attachment.id!r} exceeds 20 megapixels")
        if getattr(image, "n_frames", 1) != 1:
            image.close()
            raise ValueError(f"attachment {attachment.id!r} must be a static image")
        image.load()
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        image.close()
        if normalized.width * normalized.height > 1_048_576:
            scale = (1_048_576 / (normalized.width * normalized.height)) ** 0.5
            normalized = normalized.resize(
                (
                    max(1, int(normalized.width * scale)),
                    max(1, int(normalized.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        return normalized

    def _decode_images(self, attachments: list[Attachment]) -> list[Image.Image]:
        images: list[Image.Image] = []
        try:
            for attachment in attachments:
                images.append(self._decode_image(attachment))
            return images
        except Exception:
            for image in images:
                image.close()
            raise


_engine = SystemOneEngine()


def get_engine() -> SystemOneEngine:
    return _engine
