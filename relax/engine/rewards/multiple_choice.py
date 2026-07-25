# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re


ANS_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def extract_answer(text: str) -> str:
    text = "" if text is None else str(text)
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else text.strip()


def get_multiple_choice_reward(response, label):
    response = extract_answer(response)
    label = extract_answer(label)
    reward = 1.0 if response == label else 0.0
    return reward
