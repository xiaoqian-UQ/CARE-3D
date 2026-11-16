import pytest

from evaluator.gqa3d_metrics import summarize_gqa3d_case_obj


def test_legacy_sft_answer_markers_are_normalized(monkeypatch):
    # Keep this regression local to the scorer's public helper without pulling
    # the online evaluator into the metric implementation.
    monkeypatch.syspath_prepend("evaluator")
    from msqa_evaluator_offline import extract_cot_answer_text

    assert extract_cot_answer_text("<think_sum>x</think_sum><answer>Brown</answer>") == "Brown"
    assert extract_cot_answer_text("᠂Whiteᠬ") == "White"
    assert extract_cot_answer_text("<unused33>White<unused34>") == "White"


def test_extract_question_supports_qwen_and_gemma_chat_templates(monkeypatch):
    monkeypatch.syspath_prepend("evaluator")
    from msqa_evaluator_offline import extract_question

    assert (
        extract_question(
            "<|im_start|>user\nWhat color is it?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        == "What color is it?"
    )
    assert (
        extract_question(
            "<bos><start_of_turn>user\nWhat color is it?<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        == "What color is it?"
    )


def test_gqa3d_case_and_strict_object_score():
    scores = [100, 75, 100, 100, 50, 100]
    score_rows = [{"gpt_score": score} for score in scores]
    result_rows = [
        {"index": index, "question": f"question {index}"} for index in range(6)
    ]
    annotations = [
        {
            "question": f"question {index}",
            "scene_id": "scene0",
            "obj_ids": [index // 3],
        }
        for index in range(6)
    ]

    summary = summarize_gqa3d_case_obj(score_rows, result_rows, annotations)

    assert summary["Case"] == pytest.approx(87.5)
    assert summary["Obj."] == pytest.approx(50.0)
    assert summary["correct_object_count"] == 1
    assert summary["correct_questions_per_object"] == {"0": 0, "1": 0, "2": 1, "3": 1}


def test_gqa3d_requires_three_questions_per_object():
    with pytest.raises(ValueError, match="exactly three"):
        summarize_gqa3d_case_obj(
            [{"gpt_score": 100}],
            [{"index": 0, "question": "question"}],
            [{"question": "question", "scene_id": "scene0", "obj_ids": [1]}],
        )
