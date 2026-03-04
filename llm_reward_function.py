import json
import multiprocessing
import os
import re
from typing import Any


def _parse_scores_from_response(response_content: str, expected_len: int) -> list[float]:
    """尽可能鲁棒地从模型回复中提取分数列表。"""
    score_payload = None

    answer_match = re.search(
        r"<answer>\s*(\{.*?\})\s*</answer>", response_content, re.DOTALL
    )
    if answer_match:
        score_payload = answer_match.group(1)
    else:
        json_match = re.search(r"\{\s*\"scores\"\s*:\s*\[.*?\]\s*\}", response_content, re.DOTALL)
        if json_match:
            score_payload = json_match.group(0)

    if not score_payload:
        return [0.0] * expected_len

    try:
        score_data = json.loads(score_payload)
    except json.JSONDecodeError:
        return [0.0] * expected_len

    raw_scores = score_data.get("scores", [])
    if not isinstance(raw_scores, list) or len(raw_scores) != expected_len:
        return [0.0] * expected_len

    parsed: list[float] = []
    for s in raw_scores:
        try:
            parsed.append(max(-1.0, min(1.0, float(s))))
        except (TypeError, ValueError):
            parsed.append(0.0)
    return parsed


def _api_batch_worker(
    task_id: int,
    generated_texts: list[str],
    true_answer: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
) -> list[float]:
    """批量处理多个生成结果的 worker 函数。"""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        generated_answers_list = "\n".join(
            [f"{i + 1}. {text}" for i, text in enumerate(generated_texts)]
        )

        prompt_template = f"""You are evaluating generated answers against a gold answer in a medical QA setting.

<true_answer>
{true_answer}
</true_answer>

<generated_answers>
{generated_answers_list}
</generated_answers>

Scoring policy (strictly prioritize correctness):
1) +1.0: Semantically equivalent to the true answer (minor wording differences are okay).
2) +0.5: Partially correct; key medical intent mostly aligned but missing details.
3) -0.2: Fluent and medically plausible, but does not answer the target meaning.
4) -0.7: Contains core factual contradiction to true answer.
5) -1.0: Unreasonable output (nonsense, broken grammar, non-human-like expression).

Important constraints:
- Do NOT assign positive score to semantically incorrect answers.
- Return JSON only, no explanation.
- Output exactly {len(generated_texts)} scores.

<answer>
{{"scores": [score1, score2, ...]}}
</answer>"""

        messages_payload: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a strict reward model."},
            {"role": "user", "content": prompt_template},
        ]

        message = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            stream=False,
            messages=messages_payload,
        )
        response_content = message.choices[0].message.content or ""
        return _parse_scores_from_response(response_content, len(generated_texts))

    except Exception as e:
        print(f"[reward-worker:{task_id}] {type(e).__name__}: {e}")
        return [0.0] * len(generated_texts)


def create_llm_reward_func(
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    model_name: str = "deepseek-chat",
    temperature: float = 0.1,
    batch_size: int = 16,
    reasoning_end: str = "</think>",
    print_every_steps: int = 5,
):
    """创建 LLM 评判奖励函数。"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing API key: set api_key or OPENAI_API_KEY/DEEPSEEK_API_KEY")

    state = {"printed_times": 0}

    def check_answer_llm_optimized(prompts, completions, answer, **kwargs):
        """批处理 + 并行的 GRPO/GSPO 训练奖励函数。"""
        tasks = []
        all_extracted_responses = []
        task_id_counter = 0

        for i in range(len(prompts)):
            prompt_completions = completions[i]
            true_answer_for_prompt = answer[i]

            batch_extracted = []
            for completion in prompt_completions:
                completion_content = completion["content"]
                guess = re.search(
                    rf"{re.escape(reasoning_end)}(.*)", completion_content, re.DOTALL
                )
                extracted_response = guess.group(1).strip() if guess else completion_content
                batch_extracted.append(extracted_response)

            all_extracted_responses.append(batch_extracted)

            for j in range(0, len(batch_extracted), batch_size):
                batch = batch_extracted[j : j + batch_size]
                tasks.append(
                    (
                        task_id_counter,
                        batch,
                        true_answer_for_prompt,
                        resolved_api_key,
                        base_url,
                        model_name,
                        temperature,
                    )
                )
                task_id_counter += 1

        scores = []
        if tasks:
            try:
                cpu_count = os.cpu_count() or 1
                num_workers = min(len(tasks), cpu_count * 2)
                with multiprocessing.Pool(processes=num_workers) as pool:
                    batch_results = list(pool.starmap(_api_batch_worker, tasks))
                    for batch in batch_results:
                        scores.extend(batch)
            except Exception as e:
                print(f"FATAL: Multiprocessing pool failed: {e}")
                return [0.0] * sum(len(batch) for batch in all_extracted_responses)

        if state["printed_times"] % print_every_steps == 0 and prompts:
            print("=" * 80)
            print(f"Step {state['printed_times']} - Score sample")
            print(f"Question: {prompts[0][-1]['content'] if prompts[0] else 'No question'}")
            print(f"True Answer: {answer[0]}")
            print(f"Scores ({len(scores)}): {scores}")
            print("=" * 80)

        state["printed_times"] += 1
        return scores

    return check_answer_llm_optimized
