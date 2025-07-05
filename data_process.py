import re
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import os

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_datasets(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        'db_set':'gsm8k'
    })
    data = data.remove_columns(['question'])

    data_qa = load_dataset("qiaojin/PubMedQA", "pqa_artificial")[split]
    data_qa = data_qa.filter(lambda x: len("\n".join(x['context']['contexts'])) < 1024)
    data_qa = data_qa.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Given the scientific context below:\n" +
                          "\n".join(x['context']['contexts']) +
                          "\n\nAnswer the following question:\n" +
                          x['question'] +
                          " with 'yes', 'no' or 'maybe'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['final_decision'],
        'db_set': 'pubmedqa'
    })
    data_qa = data_qa.remove_columns(['pubid', 'question', 'context', 'long_answer', 'final_decision'])

    categories =['Lab_Medicine', 'Wearables', 'Dermatology', 'Gastroenterology', 'Internal_Medicine', 'Oncology', 'Orthopedics', 'General_Surgery', 'Ophthalmology', 'Audiology', 'Head_Neck_Surgery', 'Elderly_Care', 'Pediatrics', 'Allergy_Immunology', 'Rheumatology', 'Pharmacy', 'Obstetrics_Gynecology', 'Microbiology', 'Dentistry', 'Physical_Medicine_and_Rehabilitation', 'Neurology', 'Psychiatry', 'Pathology', 'Genetics', 'Rare_Diseases', 'Hematology', 'Emergency', 'Endocrinology', 'Radiology', 'Cardiology', 'Pulmonology', 'Infectious_Diseases', 'Critical_Care', 'Pediatric_Surgery', 'Neuroscience', 'Epidemiology', 'Fitness_Sports', 'Health_Education', 'Health_Economics', 'Health_Entrepreneurship', 'Hospital_Management', 'Mental_Health', 'Nutrition', 'Palliative_Care', 'Preventive_Medicine', 'Public_Health', 'Social_Media_Addiction', 'Sleep', 'Supplements', 'Vaccination', 'Work_Health', 'Wellbeing']
    data_mc = concatenate_datasets([load_dataset("yesilhealth/Health_Benchmarks",i)[i] for i in categories])
    data_mc = data_mc.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\nAnswer the following question:\n" +
                          x['Questions'] +
                          "\n With 'A', 'B', 'C' or 'D'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['Answers'],
        'db_set': 'med_mc'
    })
    data_mc = data_mc.remove_columns(['Answers', 'Questions'])

    dataset = concatenate_datasets([data, data_qa, data_mc])
    return dataset 

def prepare_and_save_dataset(save_dir="data_cache"):
    os.makedirs(save_dir, exist_ok=True)
    dataset = get_datasets()
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1)
    split["train"].save_to_disk(os.path.join(save_dir, "train"))
    split["test"].save_to_disk(os.path.join(save_dir, "test"))
    print(f"数据已保存到 {save_dir}/train 和 {save_dir}/test")

def load_saved_dataset(save_dir="data_cache"):
    train_dataset = load_from_disk(os.path.join(save_dir, "train"))
    test_dataset = load_from_disk(os.path.join(save_dir, "test"))
    return train_dataset, test_dataset

if __name__ == '__main__':
    save_dir = "data_cache"
    prepare_and_save_dataset(save_dir)

