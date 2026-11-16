import json

from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.scanqa_eval import ScanQAEvaluator
import torch
from data.cot_utils import parse_cot_answer, replace_cot_tokens_with_indicators

qa_type_metric_mapping_dict = {
    'appearance (grounded qa)': ['em', 'em_refined'],
    'existence (grounded qa)': ['em', 'em_refined'],
    'appearance (grounded qa)': ['em', 'em_refined'],
    'spatial (grounded qa)': ['em', 'em_refined'],
    'class (grounded qa)': ['em', 'em_refined']
}

all_metric_types = ['em', 'em_refined', 'meteor', 'rouge', 'cider', 'bleu']

@EVALUATOR_REGISTRY.register()
class GQA3DCOTEvaluator(ScanQAEvaluator):
    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em_overall': [], 'em_refined_overall': [],
            'top1_accuracy': [],
        }
        for qa_type in qa_type_metric_mapping_dict.keys():
            for metric in qa_type_metric_mapping_dict[qa_type]:
                self.eval_dict[f'{qa_type}_{metric}'] = []
        self.total_count = 0
        self.qa_type_count = {}
        for qa_type in qa_type_metric_mapping_dict.keys():
            self.qa_type_count[f'{qa_type}_count'] = 1e-10
        self.save_results = []

    def batch_metrics(self, data_dict):
        metrics = {}
        for qa_type in qa_type_metric_mapping_dict.keys():
            metrics[f'{qa_type}_count'] = 1e-10
            metrics[f'{qa_type}_em'] = 0
            metrics[f'{qa_type}_em_refined'] = 0

        em_overall = 0
        em_refined_overall = 0

        metric_qa_type_record_dict = {}
        
        for qa_type in qa_type_metric_mapping_dict.keys():
            for metric_type in all_metric_types:
                if metric_type in qa_type_metric_mapping_dict[qa_type]:
                    if metric_type not in metric_qa_type_record_dict:
                            metric_qa_type_record_dict[metric_type] = {}
                    if metric_type == 'em' or metric_type == 'em_refined':
                        metric_qa_type_record_dict[metric_type][qa_type] = 0
                    else:
                        metric_qa_type_record_dict[metric_type][qa_type] = []
            

        for answer_pred, answer_gts, qa_type in zip(
            data_dict['output_txt'], data_dict['output_gt'], data_dict['type']
        ):
            if parse_cot_answer(replace_cot_tokens_with_indicators(answer_gts[0]))['answer'] != "Empty content for this key!":
                answer_pred = parse_cot_answer(answer_pred)['answer']
                answer_gts = [parse_cot_answer(replace_cot_tokens_with_indicators(gt))['answer'] for gt in answer_gts]
            print(f'************* Evaluation *************')
            print(f'answer_gts: {answer_gts}\n')
            print(f'answer_pred: {answer_pred}\n')
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em_overall += em_flag
            em_refined_overall += em_refined_flag
            print(f'em_flag: {em_flag}\n')


            if qa_type not in qa_type_metric_mapping_dict.keys():
                continue

            metrics[f'{qa_type}_count'] += 1

            if qa_type not in ['attribute', 'description','spatial relationship']:
                metric_qa_type_record_dict['em'][qa_type] += em_flag
            
            if qa_type not in ['description']:
                metric_qa_type_record_dict['em_refined'][qa_type] += em_refined_flag

        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em_overall'] = em_overall / batch_size
        metrics['em_refined_overall'] = em_refined_overall / batch_size
        for qa_type in metric_qa_type_record_dict['em'].keys():
            metrics[f'{qa_type}_em'] = metric_qa_type_record_dict['em'][qa_type]

        for qa_type in metric_qa_type_record_dict['em_refined'].keys():    
            metrics[f'{qa_type}_em_refined'] = metric_qa_type_record_dict['em_refined'][qa_type]

        # print(f'line 100: metrics: {metrics}\n')

        metrics['target_metric'] = metrics['em_refined_overall']


        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size
        for qa_type in qa_type_metric_mapping_dict.keys():
            self.qa_type_count[f'{qa_type}_count'] += metrics[f'{qa_type}_count']

        # data_dict['pred_obj_prob'] = torch.tensor(data_dict['pred_obj_prob'])
        # pred_obj_mask = data_dict['pred_obj_prob'] > 0.5
        # gt_obj_mask = data_dict['grounding_obj_mask_gt'] > 0.5
        # if len(pred_obj_mask.shape) > len(gt_obj_mask.shape):
        #     pred_obj_mask = pred_obj_mask.squeeze(1)
        # correct_predict = torch.logical_and(pred_obj_mask, gt_obj_mask)

        for i in range(batch_size):
            self.save_results.append({
                # vision
                'source': data_dict['source'][i],
                'scene_id': data_dict['scene_id'][i],
                'position': data_dict['pos'][i],
                'orientation': data_dict['ori'][i],
                # language
                'instruction': data_dict['input_txt'][i],
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
                # 'grounding_correct': correct_predict[i].sum().item(),
            })

        # save eval dict
        for key in self.eval_dict.keys():
            if key in ['em_overall', 'em_refined_overall', 'target_metric']:
                continue
            if 'em' in key or 'em_refined' in key:
                self.eval_dict[key].append(metrics[key])
        
        for key in ['em_overall', 'em_refined_overall', 'target_metric']:
            self.eval_dict[key].append(metrics[key] * batch_size)
        
        # for key in ['top1_accuracy']:  # only record top1 acc
        #     self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):

        # em, em_refined
        print(f'record: line 159: self.eval_dict: {self.eval_dict}\n')
        for k, v in self.eval_dict.items():
            if k in ['description_cider', 'description_meteor', 'description_rouge', 'attribute_meteor', 'spatial relationship_meteor']:
                continue
            if k.split('_')[0] in qa_type_metric_mapping_dict.keys():
                self.eval_dict[k] = sum(v) / self.qa_type_count[f'{k.split("_")[0]}_count']
            else:
                self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False

        if (is_best or split == 'test') and is_main_process:
            with open(str(self.save_dir / 'results.json'), 'w') as f:
                json.dump(self.save_results, f, indent=2)

        return is_best, self.eval_dict
