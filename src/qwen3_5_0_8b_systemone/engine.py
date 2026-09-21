from __future__ import annotations

import copy
import io
import itertools
import json
import os
import string
import threading
from dataclasses import dataclass
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
    input_ids: list[int]
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
        print(f"Qwen3.5 System One batch size: {self.batch_size}")

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
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                revision=revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self.labels = self._verified_labels()

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
            answers: dict[str, Any] = {}
            offset = 0
            batch_size = max(1, self.batch_size)
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
            "You are a constrained decision scorer. Read the shared state once. "
            "For each later question, answer using exactly one supplied internal label and no other text."
        )
        state_text = f"Shared state:\n{_json(_safe(request.state))}"
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
            {"role": "assistant", "content": "State received."},
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
                {
                    "label": self.labels[index][0],
                    "option": _safe(name),
                    "description": _safe(description),
                }
                for index, (name, description) in enumerate(question.criteria.items())
            ]
        elif isinstance(question, ScoreQuestion):
            criteria = [
                {
                    "label": self.labels[index][0],
                    "level": index,
                    "description": _safe(description),
                }
                for index, description in enumerate(question.criteria)
            ]
        elif isinstance(question, NoulQuestion):
            descriptions = question.criteria or {}
            true_description = descriptions.true if descriptions else None
            false_description = descriptions.false if descriptions else None
            criteria = [
                {
                    "label": self.labels[0][0],
                    "value": True,
                    "description": _safe(
                        "yes" if true_description is None else true_description
                    ),
                },
                {
                    "label": self.labels[1][0],
                    "value": False,
                    "description": _safe(
                        "no" if false_description is None else false_description
                    ),
                },
            ]
        else:
            raise TypeError(f"unsupported question type: {type(question).__name__}")

        prompt = _json(
            {
                "question_type": question.type,
                "instructions": _safe(question.instructions),
                "criteria": criteria,
                "required_output": "one label",
            }
        )
        branch = (
            self.processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ANSWER_PREFIX
        )
        input_ids = self.processor.tokenizer.encode(branch, add_special_tokens=False)
        return CompiledQuestion(
            question_id=question_id,
            question=question,
            input_ids=input_ids,
            candidate_token_ids=[
                token_id for _, token_id in self.labels[: len(criteria)]
            ],
        )

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

        cache = copy.deepcopy(base_cache)
        cache.reorder_cache(torch.zeros(count, dtype=torch.long, device=self.device))
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
        weight = self.model.lm_head.weight
        results: list[list[float]] = []
        for row, item in enumerate(batch):
            candidate_ids = torch.tensor(item.candidate_token_ids, device=self.device)
            logits = torch.mv(
                weight.index_select(0, candidate_ids).float(), final_hidden[row]
            )
            results.append(logits.cpu().tolist())
        return results

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
