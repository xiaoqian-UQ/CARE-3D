"""One real preflight request before allocating training model state."""
import os
import time

from scenecot_verl.rewards.scenecot_reward import maybe_apply_answer_judge


if __name__ == "__main__":
    os.environ["SCENECOT_ANSWER_JUDGE_ON_ERROR"] = "raise"
    started = time.monotonic()
    score, metrics = maybe_apply_answer_judge(
        0.0, {"answer_rule_direct": 0, "answer_token_f1": 0.5},
        question="What color is the chair?", pred_answer="It is blue.", gt_answers=["blue"],
    )
    if not metrics["answer_judge_called"] or metrics["answer_judge_fallback"]:
        raise SystemExit("Judge preflight did not produce a model judgement")
    print(f"Judge preflight OK: backend={metrics['answer_judge_backend']} "
          f"model={metrics['answer_judge_model']} reward={score} "
          f"seconds={time.monotonic() - started:.2f}", flush=True)
