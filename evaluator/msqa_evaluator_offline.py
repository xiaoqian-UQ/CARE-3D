from eval_utils import load_json, execute_chat, save_to_json, path_verify
import re
import os
import torch
import json
import time
import copy
from tqdm import tqdm
from lang_eval.evaluator import Evaluator
import hydra
from omegaconf import OmegaConf
import logging
from eval_utils import parse_cot_answer, replace_cot_tokens_with_indicators
from gqa3d_metrics import summarize_gqa3d_case_obj
logger = logging.getLogger(__name__)

def extract_question(text):
    # Using regular expression to find text between 'USER:' and 'ASSISTANT:'
    match = re.search(r"USER: (.*?) ASSISTANT:", text)
    if match:
        return match.group(1)
    match = re.search(
        r"<\|im_start\|>user\s*(.*?)<\|im_end\|>\s*<\|im_start\|>assistant",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"<start_of_turn>user\s*(.*?)<end_of_turn>\s*<start_of_turn>model",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None

def extract_number(text):
    # Using regular expression to find number in text
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None

def extract_score_1_to_5(text):
    if text is None:
        return None
    for match in re.finditer(r"\b[1-5]\b", str(text)):
        return int(match.group(0))
    return None

def extract_question_type(text):
    # Pattern to match "This is a/an <type> question"
    match = re.search(r'This is (?:an?|a) (.+?) question', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return 'unknown'  # or raise an error / return 'unknown'

def extract_question_sentences(text):
    # Split text into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())

    # Look for first sentence that contains a '?'
    for i, sentence in enumerate(sentences):
        if '?' in sentence:
            return ' '.join(sentences[i:])

    # If no question mark is found, return the last sentence
    return sentences[-1] if sentences else ''

def extract_cot_answer_text(text):
    text = replace_cot_tokens_with_indicators(text)
    answer = parse_cot_answer(text)['answer']
    if answer != "Empty content for this key!":
        return answer
    # Gemma 3 SFT ground-truth strings retain the selected unused-token
    # delimiters instead of the human-readable <answer> tags.
    match = re.search(r"<unused33>(.*?)<unused34>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"᠂(.*?)ᠬ", text, re.DOTALL)
    return match.group(1).strip() if match else text

class LLMEvaluator():
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"total_cnt": 0}
        self.metric_type_list = ['gpt_score', 'em1', 'em1_strict', 'cider', 'bleu', 'meteor', 'rouge']
        for metric_type in self.metric_type_list:
            self.eval_dict[metric_type] = 0
    
    def update(self, score_dict):
        '''
            update the evaluation results
        '''
        for metric_type in score_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] += score_dict[metric_type]
        self.eval_dict["total_cnt"] += 1
    
    def summary(self):
        '''
        '''
        # self.eval_dict["gpt_score"] = self.eval_dict["gpt_score"]/self.eval_dict["total_cnt"]
        for metric_type in self.eval_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] = self.eval_dict[metric_type]/self.eval_dict["total_cnt"]
        return self.eval_dict

    def get_gpt_score(self, question, answer, gt):
        '''
            evaluate the results
        '''
        model = self.cfg.gpt_model
        api_key = self.cfg.api_key
        api_version = self.cfg.api_version
        region = self.cfg.region
        messages = copy.deepcopy(load_json(self.cfg.gpt_score_prompt_path))
        user_prompt = "\n".join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({"role": "user", "content": user_prompt})
        max_invalid_retries = int(os.environ.get("GPT_SCORE_INVALID_MAX_RETRIES", "3"))
        response = None
        for attempt in range(max_invalid_retries + 1):
            response = execute_chat(messages, api_key, model)
            score = extract_score_1_to_5(response)
            if score is not None:
                return score
            logger.warning(
                "GPT score response did not contain a 1-5 score on attempt %s/%s: %r",
                attempt + 1,
                max_invalid_retries + 1,
                response,
            )
            messages.extend(
                [
                    {"role": "assistant", "content": str(response or "")},
                    {
                        "role": "user",
                        "content": "Your previous response did not contain a valid score. Reply with only one integer from 1 to 5.",
                    },
                ]
            )

        fallback_score = int(os.environ.get("GPT_SCORE_INVALID_FALLBACK", "3"))
        fallback_score = min(5, max(1, fallback_score))
        logger.warning("Using fallback GPT score %s after invalid response: %r", fallback_score, response)
        return fallback_score

class MSQAEvaluator():
    '''
        process the data from files
    '''
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"gpt_score": 0, "total_cnt": 0}
        self.evaluator = LLMEvaluator(self.cfg)

    def eval_metrics(self):

        result_dir = self.cfg.result_dir
        dataset_names_list = self.cfg.evaluate_dataset
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        for model_name in self.cfg.evaluated_models:
            for dataset_name in dataset_names_list:
                result_score_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, f"eval_results_{file_tag}.json")
                result_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.pt") if os.path.exists(os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.pt")) else os.path.join(result_dir, model_name, "eval_results", dataset_name, "results.json")
                if not os.path.exists(result_file):
                    result_file = os.path.join(result_dir, model_name, "eval_results", dataset_name,"results.json")
                if '.pt' in result_file:
                    result_dict_list = torch.load(result_file, map_location="cpu")
                elif 'json' in result_file:
                    result_dict_list = load_json(result_file)
                
                score_list = load_json(result_score_file) if os.path.exists(result_score_file) else []
                if len(score_list) >= len(result_dict_list):
                    continue
                if score_list:
                    print(f"Resume {result_score_file} from {len(score_list)}/{len(result_dict_list)}")
                for i in tqdm(range(len(score_list), len(result_dict_list)), initial=len(score_list), total=len(result_dict_list)):
                    result_dict = result_dict_list[i]
                    if 'question' in result_dict:
                        question = result_dict["question"]
                    else:
                        if "instruction" in result_dict:
                            question = extract_question(result_dict["instruction"])  
                        elif "prompt" in result_dict:
                            question = result_dict["prompt"]
                    if "response_gt" in result_dict:
                        gt = result_dict["response_gt"][0]
                    elif "gt_answer" in result_dict:
                        gt = result_dict["gt_answer"]
                    if "response_pred" in result_dict:
                        answer = result_dict["response_pred"]
                    elif 'text' in result_dict:
                        answer = result_dict["text"]

                    question = extract_question_sentences(question)
                    # Normalize both current tags and legacy model-specific
                    # action tokens before sending only the final answer to GPT.
                    answer = extract_cot_answer_text(answer)
                    # Older SFT result files wrap ground truth as ᠂...ᠬ.
                    # Normalize it independently of the prediction format so
                    # SFT and verl checkpoints receive the same GPT rubric input.
                    gt = extract_cot_answer_text(gt)
                    scored_dict = {"question": question, "answer": answer, "gt": gt}
                    if 'index' in result_dict:
                        if type(result_dict['index']) == str:
                            index = int(result_dict['index'])
                        elif type(result_dict['index']) == int:
                            index = result_dict['index']
                        else:
                            index = int(result_dict['index'].item())
                    else:
                        index = None
                    if 'type' in result_dict:
                        scored_dict['type'] = result_dict['type']
                    elif 'question_type' in result_dict:
                        scored_dict['type'] = result_dict['question_type']
                    elif 'type' not in scored_dict:
                        scored_dict['type'] = extract_question_type(answer)
                    print("question: ", question)
                    print("answer: ", answer)
                    print("gt: ", gt)
                    if 'with_gpt_score' in result_score_file:
                        score = self.evaluator.get_gpt_score(question, answer, gt)
                        scored_dict['gpt_score'] = (score-1)*25
                        print("gpt score: ", scored_dict['gpt_score'])
                        print(f"\n")
                        sleep_after_gpt_score = float(os.environ.get("GPT_SCORE_SLEEP", "0"))
                        if sleep_after_gpt_score > 0:
                            time.sleep(sleep_after_gpt_score)
                    lang_evaluator = Evaluator()
                    scored_dict.update(lang_evaluator.eval_instance(answer, [gt]))
                    self.evaluator.update(scored_dict)
                    if index:
                        scored_dict['index'] = index
                    score_list.append(scored_dict)
                    if 'with_gpt_score' in result_score_file:
                        save_to_json(result_score_file, score_list)
                summary_dict = self.evaluator.summary()
                save_to_json(result_score_file, score_list)

        # GQA3D uses three QA cases per target object. Its official-style
        # summary is Case (mean GPT score) and Obj. (all three cases correct).
        is_gqa3d = str(self.cfg.get('evaluation_protocol', '')).lower() == 'gqa3d' or 'gqa3d' in str(result_dir).lower()
        if file_tag == 'with_gpt_score' and is_gqa3d:
            annotation_path = str(self.cfg.gqa3d_annotation_path)
            annotation_rows = load_json(annotation_path)
            for model_name in self.cfg.evaluated_models:
                summaries = {}
                for dataset_name in dataset_names_list:
                    dataset_dir = os.path.join(result_dir, model_name, "eval_results", dataset_name)
                    result_score_file = os.path.join(dataset_dir, f"eval_results_{file_tag}.json")
                    result_file = os.path.join(dataset_dir, "results.pt")
                    if os.path.exists(result_file):
                        result_rows = torch.load(result_file, map_location="cpu")
                    else:
                        result_file = os.path.join(dataset_dir, "results.json")
                        result_rows = load_json(result_file)
                    score_rows = load_json(result_score_file)
                    summaries[dataset_name] = summarize_gqa3d_case_obj(
                        score_rows,
                        result_rows,
                        annotation_rows,
                        correct_threshold=float(self.cfg.gqa3d_obj_correct_threshold),
                    )

                save_path = os.path.join(result_dir, model_name, 'eval_results')
                path_verify(save_path)
                output_path = os.path.join(save_path, f"GQA3D_eval_results_{file_tag}.json")
                save_to_json(output_path, summaries)
                for dataset_name, summary in summaries.items():
                    print(
                        f"GQA3D {dataset_name}: Case={summary['Case']:.4f}, "
                        f"Obj.={summary['Obj.']:.4f} "
                        f"({summary['correct_object_count']}/{summary['object_count']})"
                    )
                print(f"Wrote {output_path}")
            return
        
        dataset_names_list = self.cfg.evaluate_dataset
        testset_mapping_dict = load_json(self.cfg.testset_mapping_path)
        QA_type_list = [
            "counting",
            "existence",
            "attribute",
            "spatial relationship",
            "navigation",
            "refer",
            "affordance",
            "description",
            "room type",
        ]
        # statistic_dict = {'scannet': {}, 
        #                  'RScan': {},
        #                  'ARKitScenes': {}}
        statistic_dict = {}
        for dataset in dataset_names_list:
            statistic_dict[dataset] = {}
            
        testset_mapping_dict = load_json(self.cfg.testset_mapping_path)
        # metric_type_list = ['em1', 'em1_strict']
        metric_type_list = []
        result_dict = {}
        result_dir = self.cfg.result_dir

        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        if file_tag == 'with_gpt_score':
            metric_type_list.append('gpt_score')
        for model_name in self.cfg.evaluated_models:
            for dataset_name in dataset_names_list:
                result_score_file = os.path.join(result_dir, model_name, "eval_results", dataset_name, f"eval_results_{file_tag}.json")
                if not os.path.exists(result_score_file):
                    result_score_file = result_score_file.replace("/scannet", "/QACOTScanNetMSR3D")
                scores_data = load_json(result_score_file)
                data_instance_cnt = 0
                for data_instance in scores_data:
                    if 'type' in data_instance:
                        data_QA_type = data_instance['type']
                    else:
                        data_QA_type = testset_mapping_dict[dataset_name][str(data_instance['index'])]['image_text']['type']
                    for metric_type in metric_type_list:
                        if metric_type not in statistic_dict[dataset_name]:
                            statistic_dict[dataset_name][metric_type] = {}
                        for QA_type in QA_type_list:
                            if QA_type in data_QA_type.lower():
                                if QA_type not in statistic_dict[dataset_name][metric_type]:
                                    statistic_dict[dataset_name][metric_type][QA_type] = {'score': [], 'cnt': 0, "avg": 0}
                                if metric_type not in data_instance:
                                    statistic_dict[dataset_name][metric_type][QA_type]['score'].append((data_instance['score']-1)*25)
                                else:
                                    statistic_dict[dataset_name][metric_type][QA_type]['score'].append(data_instance[metric_type])
                                statistic_dict[dataset_name][metric_type][QA_type]['cnt'] += 1

            for dataset_name in dataset_names_list:
                for metric_type in metric_type_list:
                    for QA_type in QA_type_list:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            statistic_dict[dataset_name][metric_type][QA_type]['avg'] = sum(statistic_dict[dataset_name][metric_type][QA_type]['score'])/statistic_dict[dataset_name][metric_type][QA_type]['cnt']
            
            for metric_type in metric_type_list:
                statistic_dict[metric_type] = {'overall': {}}
                for QA_type in QA_type_list:
                    score_list = []
                    cnt = 0
                    for dataset_name in dataset_names_list:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                    if cnt > 0:
                        statistic_dict[metric_type]['overall'][QA_type] = sum(score_list)/cnt

            merged_QA_type_list = ['counting', 'existence', 'attribute_description', 'spatial_refer', 'navigation', 'others']
            for metric_type in metric_type_list:
                statistic_dict[metric_type]['merged'] = {}

            for metric_type in metric_type_list:
                for QA_type in merged_QA_type_list:
                    score_list = []
                    cnt = 0
                    for dataset_name in dataset_names_list:
                        if QA_type in ['counting', 'existence', 'navigation']:
                            if QA_type in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                        elif QA_type == 'attribute_description':
                            if 'attribute' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['attribute']['avg'] * statistic_dict[dataset_name][metric_type]['attribute']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['attribute']['cnt']
                            if 'description' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['description']['avg'] * statistic_dict[dataset_name][metric_type]['description']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['description']['cnt']
                        elif QA_type == 'spatial_refer':
                            if 'spatial relationship' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['spatial relationship']['avg'] * statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt']
                            if 'refer' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['refer']['avg'] * statistic_dict[dataset_name][metric_type]['refer']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['refer']['cnt']
                        elif QA_type == 'others':
                            if 'affordance' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['affordance']['avg'] * statistic_dict[dataset_name][metric_type]['affordance']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['affordance']['cnt']
                            if 'room type' in statistic_dict[dataset_name][metric_type]:
                                score_list.append(statistic_dict[dataset_name][metric_type]['room type']['avg'] * statistic_dict[dataset_name][metric_type]['room type']['cnt'])
                                cnt += statistic_dict[dataset_name][metric_type]['room type']['cnt']
                        else:
                            ValueError("Invalid QA type")
                    if cnt > 0:
                        statistic_dict[metric_type]['merged'][QA_type] = sum(score_list)/cnt
                        statistic_dict[metric_type]['merged'][QA_type + '_cnt'] = cnt
                available_QA_types = [
                    QA_type for QA_type in merged_QA_type_list
                    if QA_type in statistic_dict[metric_type]['merged']
                    and QA_type + '_cnt' in statistic_dict[metric_type]['merged']
                ]
                if available_QA_types:
                    statistic_dict[metric_type]['merged']['weighted_avg_score'] = sum(
                        statistic_dict[metric_type]['merged'][QA_type]
                        * statistic_dict[metric_type]['merged'][QA_type + '_cnt']
                        for QA_type in available_QA_types
                    ) / sum(
                        statistic_dict[metric_type]['merged'][QA_type + '_cnt']
                        for QA_type in available_QA_types
                    )
                result_dict = {}
                # for key in statistic_dict['em1']['merged']:
                #     if 'cnt' in key:
                #         continue
                #     if 'weighted' in key:
                #         result_dict['EM-R_overall'] = statistic_dict['em1']['merged'][key]
                #     else:
                #         result_dict[f'EM-R_{key}'] = statistic_dict['em1']['merged'][key]
                if 'gpt_score' in statistic_dict:
                    for key in statistic_dict['gpt_score']['merged']:
                        if 'cnt' in key:
                            continue
                        if 'weighted' in key:
                            result_dict['GPT-Score_overall'] = statistic_dict['gpt_score']['merged'][key]
                        else:
                            result_dict[f'GPT-Score_{key}'] = statistic_dict['gpt_score']['merged'][key]
            save_path = os.path.join(result_dir, model_name, 'eval_results')
            path_verify(save_path)
            
            save_to_json(os.path.join(save_path, f"MSQA_eval_results_{file_tag}.json"), statistic_dict)

def evaluate_execute(cfg):
    '''
        execute the evaluation
    '''
    msqaevaluator = MSQAEvaluator(cfg)
    msqaevaluator.eval_metrics()

@hydra.main(version_base=None, config_path="msqa", config_name="configs")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    evaluate_execute(cfg) 

if __name__ == "__main__":
    main()
