from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_PATH = Path(str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/ljy/models/Qwen3-VL-2B-Instruct"))
TASKS = ["approach", "align", "close_gripper", "lift", "transport", "place"]


def _json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"Qwen did not return JSON: {text[:300]}")
        return json.loads(match.group(0))


class LocalQwenPlanner:
    """Persistent, API-free Qwen3-VL planner running on a MUSA device."""

    def __init__(self, model_path: Path = MODEL_PATH, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device).eval()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve physical-vs-logical MUSA numbering after visibility masks."""
        if requested == "auto":
            if hasattr(torch, "musa") and torch.musa.is_available():
                return "musa:0"
            return "cpu"
        if requested.startswith("musa:") and hasattr(torch, "musa"):
            count = torch.musa.device_count()
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                return requested
            if index >= count and count == 1:
                print(f"[Qwen] {requested} is not a visible logical device; "
                      "using musa:0 instead.", flush=True)
                return "musa:0"
        return requested

    def route_task(self, instruction: str, image: Path | None = None) -> dict:
        """One-shot task routing.  Qwen is not used again during execution."""
        prompt = f"""Classify the user's robot task.
Instruction: {instruction}
Supported task: grasp_block (reach for the wooden block, grasp it, and lift it).
Everything else is unsupported. Return JSON only:
{{"task":"grasp_block","confidence":0.0,"reason":"short"}}
The task value must be exactly grasp_block or exactly unsupported. The vertical bar is not part of a value."""
        content = []
        if image is not None:
            content.append({"type": "image", "image": Image.open(image).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": (
                "You are only a startup task router. Do not plan motions, coordinates, "
                "stages, or control commands. Return JSON only.")},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = _json_object(answer)
        raw_task = str(result.get("task", "unsupported")).strip().lower()
        # Small models sometimes repeat the schema alternatives literally,
        # e.g. "grasp_block|supported".  Normalize that unambiguous answer.
        task = "grasp_block" if raw_task.startswith("grasp_block") else "unsupported"
        instruction_lower = instruction.lower()
        explicit_grasp = (any(word in instruction_lower for word in
                              ("grasp", "pick up", "pickup", "grab")) and
                          any(word in instruction_lower for word in
                              ("block", "wooden", "cube", "wood")))
        if task == "unsupported" and explicit_grasp:
            task = "grasp_block"
            result["reason"] = (str(result.get("reason", "")) +
                                "; normalized from explicit grasp-block instruction")
        return {"task": task,
                "confidence": float(np.clip(result.get("confidence", 0.0), 0, 1)),
                "reason": str(result.get("reason", "")),
                "raw": answer, "model": self.model_path.name}

    def decide_stage(self, image: Path, stage: str, state: dict) -> dict:
        prompt = f"""Current stage: {stage}.
MuJoCo state (world frame, metres): {json.dumps(state, ensure_ascii=False)}
Judge the image and state. Choose the next semantic action. Numeric telemetry is authoritative.
Before GRASP, aligned may be true only when the grasp center is inside the calibrated gripper
capture envelope and the hand is not pushing the block away. If not aligned, request exactly
one correction direction. Do not output joint angles, torque, or coordinates in text.
Return JSON only:
{{"assessment":"short", "aligned":false, "correction_direction":"left|right|up|down|forward|backward|none", "next_text":"short ARDY-style action without numbers", "confidence":0.0}}"""
        messages = [
            {"role": "system", "content": "You are the visual high-level planner for a Unitree G1 grasp. Return JSON only."},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(image).convert("RGB")},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=140, do_sample=False)
        answer = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        result = _json_object(answer)
        result["raw"] = answer
        result["model"] = self.model_path.name
        return result

    def rewrite_fixed_stage_prompt(self, image: Path, stage: str,
                                   fixed_prompt: str, state: dict) -> dict:
        """Lightly qualify a proven ARDY prompt without changing its action."""
        compact = {key: state[key] for key in
                   ("block_world_m", "grasp_center_world_m", "contact_groups")
                   if key in state}
        prompt = f"""Stage: {stage}
Reference ARDY prompt: {fixed_prompt}
Current simulator state: {json.dumps(compact)}
Look at the image and select exactly one mild visual qualifier for the reference action.
Choose from carefully, steadily, gently, firmly, securely, or slowly. Do not rewrite or repeat
the reference prompt, add coordinates, change objects/directions, or introduce another stage.
Return JSON only: {{"qualifier":"word","visual_note":"short","confidence":0.0}}"""
        messages = [
            {"role": "system", "content": (
                "You only make tiny wording adjustments to a proven robot-action prompt. "
                "Never change its action or stage. Return JSON only.")},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(image).convert("RGB")},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            result = _json_object(answer)
        except (ValueError, json.JSONDecodeError):
            # A malformed visual note must never stop the proven grasp path.
            # Qualifier recovery below can still use the model's raw text.
            result = {}
        candidate = str(result.get("ardy_prompt", fixed_prompt)).strip()
        # Hard guardrail: Qwen may only add a little wording, never replace the
        # proven prompt with a semantically different command.
        reference_words = {word.strip(".,").lower() for word in fixed_prompt.split()
                           if len(word.strip(".,")) >= 4}
        candidate_words = {word.strip(".,").lower() for word in candidate.split()}
        overlap = len(reference_words & candidate_words) / max(1, len(reference_words))
        if (overlap < .65 or len(candidate.split()) > len(fixed_prompt.split()) + 5
                or ("person" in fixed_prompt.lower() and "person" not in candidate.lower())):
            candidate = fixed_prompt
        # Preserve the proven action verbatim, but let the live visual result
        # contribute one harmless conditioning word.  This avoids accepting a
        # small VLM's accidental stage/object changes while ensuring every
        # stage prompt is actually image-conditioned rather than decorative.
        allowed = ("carefully", "steadily", "gently", "firmly", "securely", "slowly")
        qualifier = str(result.get("qualifier", "")).strip().lower().strip(".,")
        if qualifier not in allowed:
            visual_words = (answer + " " + str(result.get("visual_note", ""))).lower().replace(",", " ").split()
            qualifier = next((word.strip(".") for word in visual_words
                              if word.strip(".") in allowed), "")
        # Do not duplicate a qualifier that is already present in the proven
        # sentence.  Repeating e.g. "slowly" changed ARDY's diffusion result
        # enough to miss the block even though the requested action was the
        # same.  Qwen still makes the per-stage visual decision and we log it,
        # but the exact successful wording is preserved when it already
        # expresses that decision.
        qualifier_already_present = bool(
            qualifier and re.search(rf"\b{re.escape(qualifier)}\b", fixed_prompt, re.I)
        )
        conditioned = fixed_prompt
        if qualifier and not qualifier_already_present:
            conditioned += f" Continue {qualifier}."
        return {"ardy_prompt": conditioned,
                "qualifier": qualifier,
                "qualifier_already_present": qualifier_already_present,
                "visual_note": str(result.get("visual_note", "")),
                "confidence": float(np.clip(result.get("confidence", .5), 0, 1)),
                "raw": answer, "model": self.model_path.name}

    def decide_visual_trace(self, image: Path, state: dict) -> dict:
        """Return task progress and the next image-space end-effector trace."""
        compact_state = {key: state[key] for key in ("contacts", "holding", "lifted")
                         if key in state}
        prompt = f"""Plan the next short right-hand action for grasping the block.
State: {json.dumps(compact_state)}
Image is a crop. Predict endpoint e=[x,y] using Qwen visual-grounding coordinates:
top-left is [0,0], bottom-right is [1000,1000], x grows right and y grows down.
The manipulation target is the RED block on the right side of the table; ignore the yellow block.
For approach or align, e must mark the centre of that visible red block, not the crop centre
unless the red block is actually centred. Inspect the image; do not copy example coordinates.
Use only task IDs: approach, align, close_gripper, lift, transport, place.
p is a short action label for logging only. Return JSON only with exactly the keys
d (completed IDs), r (remaining IDs), n (next ID), p, e (two floats), and c (confidence)."""
        messages = [
            {"role": "system", "content": (
                "You are a rolling visual task planner for a Unitree G1. "
                "Never mark a subtask complete unless the image and telemetry support it. "
                "Return one valid JSON object only.")},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(image).convert("RGB")},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        answer = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = _json_object(answer)
        endpoint = result.get("e")
        if not isinstance(endpoint, list) or len(endpoint) != 2:
            raise ValueError("Qwen visual trace endpoint is invalid")
        completed = [item for item in result.get("d", []) if item in TASKS]
        remaining = [item for item in result.get("r", []) if item in TASKS and item not in completed]
        # Always expose an explicit partition even if the small model omitted it.
        if not remaining:
            remaining = [item for item in TASKS if item not in completed]
        return {
            "completed_subtasks": completed,
            "remaining_subtasks": remaining,
            "next_subtask": str(result.get("n", "align")),
            "ardy_prompt": str(result.get("p", "carefully align the right hand")),
            "trace": {"coordinate_space": "crop_normalized",
                      "end_uv": [float(np.clip(value / 1000., 0, 1))
                                 for value in endpoint]},
            "confidence": float(np.clip(result.get("c", .5), 0, 1)),
            "raw": answer, "model": self.model_path.name}
