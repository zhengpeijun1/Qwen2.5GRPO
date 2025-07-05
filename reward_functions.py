import re
from data_process import extract_xml_answer

def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    rewards = []
    for r,a,dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(1.0)
            elif r == a:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)
    return rewards

def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r,dt in zip(extracted_responses,db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(0.5 if ('yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
        else:
            rewards.append(0.5 if ('a' in r.lower() or 'b' in r.lower() or 'c' in r.lower() or 'd' in r.lower()) else 0.0)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents] 

def safety_reward_func(completions, **kwargs) -> list[float]:
    dangerous_keywords = [
        "stop medication", "stop taking your medicine", "ignore doctor's advice", "double the dose", "overdose",
        "self-medicate", "untested treatment", "miracle cure", "no side effects", "cure cancer instantly",
        "replace your doctor", "ignore side effects", "no need for diagnosis", "guaranteed cure",
        "随意停药", "自行加量", "无副作用", "包治百病", "偏方治愈", "不需要医生", "忽略医生建议", "自行用药"
    ]
    fake_medical_claims = [
        "cure all diseases", "permanent cure", "miracle drug", "100% effective", "no risk at all",
        "包治百病", "永久治愈", "神奇药物", "百分百有效", "完全没有风险"
    ]
    forbidden_words = [
        "drugs","prohibited drugs","illegal drugs","date rape drug","hallucinogen","drug trafficking",
        "毒品", "违禁药物", "非法药物", "毒蘑菇", "迷奸药", "致幻剂", "毒品交易"
    ]
    all_bad_phrases = dangerous_keywords + fake_medical_claims + forbidden_words
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        r_lower = r.lower()
        found = False
        for phrase in all_bad_phrases:
            if phrase.lower() in r_lower:
                found = True
                break
        rewards.append(-1.0 if found else 0.0)
    return rewards 