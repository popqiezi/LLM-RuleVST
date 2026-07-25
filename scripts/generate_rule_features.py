#!/usr/bin/env python3
"""Generate and validate rule features using Qwen3-4B."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from validate_output import validate_output

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_user_prompt(template: str, case: dict[str, Any], xml_text: str) -> str:
    rel = case["relative_navigation_state"]
    own = case["own_ship"]
    target = case["target_ship"]
    return template.format(
        retrieved_xml_rules=xml_text,
        waterway_type=case["waterway_type"],
        own_ship_id=own["ship_id"],
        own_ship_type=own["ship_type"],
        own_ship_length_m=own["length_m"],
        target_ship_id=target["ship_id"],
        target_ship_type=target["ship_type"],
        target_ship_length_m=target["length_m"],
        historical_state_window=json.dumps(case["historical_state_window"], ensure_ascii=False),
        range_m=rel["range_m"],
        relative_bearing_deg=rel["relative_bearing_deg"],
        relative_course_difference_deg=rel["relative_course_difference_deg"],
        relative_speed_mps=rel["relative_speed_mps"],
        dcpa_m=rel["DCPA_m"],
        tcpa_s=rel["TCPA_s"],
    )


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs/qwen3_rule_reasoning.yaml").read_text(encoding="utf-8"))
    llm_cfg = config["llm_rule_reasoning"]
    seed = int(llm_cfg["decoding"]["random_seed"])
    set_seed(seed)

    case = json.loads(args.case.read_text(encoding="utf-8"))
    vocab = json.loads((ROOT / llm_cfg["resources"]["vocabulary"]).read_text(encoding="utf-8"))
    system_prompt = load_text(ROOT / llm_cfg["resources"]["system_prompt"])
    user_template = load_text(ROOT / llm_cfg["resources"]["user_prompt_template"])
    xml_text = load_text(ROOT / llm_cfg["resources"]["xml_rule_base"])
    user_prompt = build_user_prompt(user_template, case, xml_text)

    model_id = llm_cfg["model"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(llm_cfg["model"]["max_new_tokens"]),
            num_beams=1,
            repetition_penalty=1.0,
        )

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        output = extract_json_object(raw_text)
        errors = validate_output(output, vocab, case["retrieved_candidate_rules"])
    except (ValueError, json.JSONDecodeError) as exc:
        output = None
        errors = [f"E01_JSON_PARSE_ERROR:{exc}"]

    record = {
        "case_id": case["case_id"],
        "seed": seed,
        "raw_output": raw_text,
        "parsed_output": output,
        "valid": not errors,
        "errors": errors,
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
