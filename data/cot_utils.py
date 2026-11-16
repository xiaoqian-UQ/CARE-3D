import os
import re

from data.data_utils import VICUNA_ACTION_TOKENS

QWEN_VL_ACTION_TOKENS = {
    token: token_id for token, token_id in zip(
        [
            '롑', '믑', '뻅', '뼝', '섐', '쉡', '싲', '쏱', '엤', '읩',
            '읿', '쟙', '젰', '쥉', '튭', '핮', 'ﮏ', '🅱', '🆒', '🕋',
            'ɘ', 'ʓ', 'Ճ', 'ഴ', 'ཅ', 'ᆺ', 'ሊ', 'ረ', 'ሾ', 'ቐ',
            'ጃ', 'ጽ', 'ᔭ', '᠂', 'ᠬ',
        ],
        range(151435, 151470),
    )
}


def select_gemma3_action_tokens(tokenizer, count=35):
    """Select stable, unused Gemma tokens for SceneCOT control actions."""
    unused_pattern = re.compile(r"<unused(\d+)>")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    candidates = []
    for token, token_id in tokenizer.get_vocab().items():
        match = unused_pattern.fullmatch(token)
        if match is None or token_id in special_ids:
            continue
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids == [token_id]:
            candidates.append((int(match.group(1)), token, int(token_id)))

    candidates.sort()
    if len(candidates) < count:
        raise ValueError(
            f"Gemma 3 tokenizer exposes only {len(candidates)} usable <unusedN> "
            f"tokens; SceneCOT requires {count}"
        )
    selected = candidates[:count]
    return {token: token_id for _, token, token_id in selected}


def load_gemma3_action_tokens():
    model_path = os.environ.get("SCENECOT_LLM_PATH", "").strip()
    if not model_path:
        raise ValueError(
            "SCENECOT_LLM_PATH is required when SCENECOT_COT_TOKEN_SET=gemma3"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return select_gemma3_action_tokens(tokenizer)

COT_INDICATORS = {
    "think_type": ["<think_type>", "</think_type>"],
    "think_grd": ["<think_grd>", "</think_grd>"],
    "think_rgn": ["<think_rgn>", "</think_rgn>"],
    "OBJ": ["[OBJ]"],
    "think_task": ["<think_task>", "</think_task>"],
    "list_obj_prob": ["<list_obj_prob>"],
    "list_obj_loc_prob": ["<list_obj_loc_prob>"],
    "list_rgn_obj": ["<list_rgn_obj>"],
    "highlight_obj": ["<highlight_obj>"],
    "img_token_indicator": ["<img_start>", "<img_end>"],
    "obj_prob": ["<obj_prob>", "</obj_prob>"],
    "obj_cap": ["<obj_cap>", "</obj_cap>"],
    "obj_loc_prob": ["<obj_loc_prob>", "</obj_loc_prob>"],
    "obj_loc_plr_prob": ["<obj_loc_plr_prob>", "</obj_loc_plr_prob>"],
    "list_obj_loc_plr_prob": ["<list_obj_loc_plr_prob>"],
    "think_sum": ["<think_sum>", "</think_sum>"],
    "answer": ["<answer>", "</answer>"],
}

COT_INDICATORS_LIST = []
for k,v in COT_INDICATORS.items():
    COT_INDICATORS_LIST.extend(v)

_cot_token_set = os.environ.get("SCENECOT_COT_TOKEN_SET", "").lower()
if not _cot_token_set and "qwen" in os.environ.get("SCENECOT_LLM_PATH", "").lower():
    _cot_token_set = "qwen"
if _cot_token_set == "qwen":
    ACTION_TOKENS = QWEN_VL_ACTION_TOKENS
elif _cot_token_set == "gemma3":
    ACTION_TOKENS = load_gemma3_action_tokens()
else:
    ACTION_TOKENS = VICUNA_ACTION_TOKENS

COT_INDICATORS_TOKENIZE = {
    k: v for k,v in zip(COT_INDICATORS_LIST, list(ACTION_TOKENS.keys())[-len(COT_INDICATORS_LIST):])
}

COT_INDICATORS_DETOKENIZE = {
    v: k for k,v in zip(COT_INDICATORS_LIST, list(ACTION_TOKENS.keys())[-len(COT_INDICATORS_LIST):])
}

GRD_TOKEN_TXT = COT_INDICATORS_TOKENIZE[COT_INDICATORS['OBJ'][0]]
OBJ_PROB_START_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_prob>']]
OBJ_PROB_END_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_prob>']]
OBJ_LOC_START_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_loc_prob>']]
OBJ_LOC_END_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_loc_prob>']]
OBJ_LOC_PLR_START_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_loc_plr_prob>']]
OBJ_LOC_PLR_END_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_loc_plr_prob>']]
LIST_OBJ_PROB_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_prob>']]
LIST_OBJ_LOC_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_loc_prob>']]
LIST_OBJ_LOC_PLR_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_loc_plr_prob>']]
LIST_RGN_OBJ_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_rgn_obj>']]
HIGHLIGHT_OBJ_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<highlight_obj>']]
IMG_START_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<img_start>']]
IMG_END_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<img_end>']]
GRD_TXT_LEFT_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<think_grd>']]
GRD_TXT_RIGHT_TOKEN_ID = ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</think_grd>']]

def parse_cot_answer(text):
    parsed_data = {}
    
    for key, markers in COT_INDICATORS.items():
        start_marker, end_marker = markers[0], markers[-1]
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            parsed_data[key] = match.group(1).strip()
        elif len(markers) == 1 and start_marker in text:  # Single-token indicators like "[OBJ]"
            parsed_data[key] = True  # Presence flag
        else:
            parsed_data[key] = "Empty content for this key!"  # fill with empty string if matching not found

    return parsed_data

def replace_cot_indicators_with_tokens(text):   # used in data/datasets.py
    for indicator, token in COT_INDICATORS_TOKENIZE.items():
        text = text.replace(indicator, token)
    return text

def replace_cot_tokens_with_indicators(text):   # used in model/leo_cot_agent.py
    for indicator, token in COT_INDICATORS_TOKENIZE.items():
        text = text.replace(token, indicator)
    return text
