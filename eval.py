import torch
import pickle
import torch.nn.functional as F
import nltk
from nltk import ngrams
import os
from neptune.utils import stringify_unsupported
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import abc
from tqdm import tqdm
from pdb import set_trace
import hydra

# from model import create_tokenizer
# from utils import set_random_seed
# from trainer import *
# from data_loader import make_pytorch_dataset, CollateForOKT, get_lstm_inputs, read_data
# from data_loader import build_prompt_with_special_tokens
# from evaluator.CodeBLEU import calc_code_bleu

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_accuracy(model, input_set, configs):
    # model.eval()
    correct = 0
    total = 0
    i = 0
    target_list = []
    predicted_list = []
    dist_list = []
    with torch.no_grad():
        for index, row in input_set.iterrows():
            A1 = row['code_i_1']
            A2 = row['code_i_2']
            B1 = row['code_j_1']
            B2 = row['code_j_2']
            target = row['is_similar']
            outputs = model.forward(A1, A2, B1, B2)
            # output = torch.round(torch.sigmoid(output))
            output1, output2 = outputs
            # print(output1, output2)
            dist = F.pairwise_distance(output1, output2)
            if dist < configs.margin:
                predicted = 1
            else:
                predicted = 0
            target_list.append(target)
            predicted_list.append(predicted)
            dist_list.append(dist.cpu().item())
            total += 1
            if (target == True and predicted == 1) or (target == False and predicted == 0):
                correct += 1
            # if i % 500 == 0: print(correct, total, predicted, target)
            i += 1
    print(target_list)
    print(predicted_list)
    dist_list = [f"{dist:.3f}" for dist in dist_list]
    print(dist_list)
    return correct / total


def get_model_accuracy(configs, model, train_set, test_set, valid_set):
    train_accuracy = get_accuracy(model, train_set, configs)
    print(f'Train accuracy: {train_accuracy}')
    test_accuracy = get_accuracy(model, test_set, configs)
    print(f'Test accuracy: {test_accuracy}')
    valid_accuracy = get_accuracy(model, valid_set, configs)
    print(f'Valid accuracy: {valid_accuracy}')

    return train_accuracy, test_accuracy, valid_accuracy

# def evaluate(configs, now, test_set, lstm_inputs, tokenizer, run, student_id_to_index, device):
#     # TODO p2: add batch generation for GPT2 and assert outputs are equal to single generation (https://github.com/huggingface/transformers/pull/7552)
#     results = {}
#     lstm = None
#     student_params_h_bar_static = None
#     student_params_h_hat_mu = None
#     student_params_h_hat_sigma = None
#     student_params_h_hat_discrete = None
#     if configs.save_model:
#         # Load best models
#         model = torch.load(os.path.join(configs.model_save_dir, now, 'model'))
#         linear = torch.load(os.path.join(configs.model_save_dir, now, 'linear'))    
#         if configs.use_lstm:
#             lstm = torch.load(os.path.join(configs.model_save_dir, now, 'lstm'))
#         elif( configs.use_h_bar_static ):
#             student_params_h_bar_static = torch.load(os.path.join(configs.model_save_dir, now, 'student_params_h_bar_static'))            
#         if configs.use_q_model:
#             # We don't require Q model for generation, only per student h hat distribution params                     
#             if( configs.dim_normal > 0 ):
#                 student_params_h_hat_mu = torch.load(os.path.join(configs.model_save_dir, now, 'student_params_h_hat_mu'))     
#                 student_params_h_hat_sigma = torch.load(os.path.join(configs.model_save_dir, now, 'student_params_h_hat_sigma'))
#             if( configs.dim_categorical > 0 ):
#                 student_params_h_hat_discrete = torch.load(os.path.join(configs.model_save_dir, now, 'student_params_h_hat_discrete'))
    
#     # Set model to eval mode
#     model.eval()
#     linear.eval()
#     if configs.use_lstm:
#         lstm.eval()

#     test_set = make_pytorch_dataset(test_set, None, do_lstm_dataset=False)

#     generated_codes = []
#     ground_truth_codes = []
#     prompts = []
#     for idx in tqdm(range(len(test_set)), desc="inference"):
#         generated_code, ground_truth_code, prompt = generate_code(test_set, lstm_inputs, tokenizer, idx, model, lstm, linear, configs, 
#                                                                        student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, 
#                                                                        student_params_h_hat_discrete, student_id_to_index, device)
#         generated_codes.append(generated_code)
#         ground_truth_codes.append(ground_truth_code)
#         prompts.append(prompt)
    
#     ## compute codebleu
#     codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_codes, generated_codes)
#     results['codebleu'] = codebleu_score
#     results['detailed_codebleu'] = detailed_codebleu_score
    
#     ## compute diversity
#     metrics = {'dist_1': Distinct_N(1), 
#                'dist_2': Distinct_N(2), 
#                'dist_3': Distinct_N(3),
#     }
#     for i, (name, metric) in enumerate(metrics.items()):
#         metric_result = metric.compute_metric(generated_codes)
#         results[name] = metric_result

#     print(f"results: {results}")

#     ## save results
#     results['generated_codes'] = generated_codes
#     results['ground_truth_codes'] = ground_truth_codes
#     results['prompts'] = prompts
#     if configs.save_model:
#         with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
#             pickle.dump(results, f)

#     ## write results to neptune
#     if configs.use_neptune:  
#         for idx, (k, v) in enumerate(results.items()):
#             run['metrics/test/generation_{}'.format(k)] = stringify_unsupported(v)


# def generate_code(test_set, lstm_inputs, tokenizer, idx, model, lstm, linear, configs, student_params_h_bar_static, 
#                 student_params_h_hat_mu, student_params_h_hat_sigma, student_params_h_hat_discrete, student_id_to_index, device):
#     # Get student knowledge state
#     student, step, prompt, code = test_set[idx]['SubjectID'], test_set[idx]['step'], test_set[idx]['next_prompt'], test_set[idx]['next_code']
#     student_id = torch.LongTensor( [ student_id_to_index[student] ] ).to(device)
#     ks, _, _ = get_knowledge_states_for_generator(lstm, lstm_inputs, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, student_params_h_hat_discrete, [student], [step], configs, student_id, device, generation=True)
    
#     # Get generator input
#     inputs = tokenizer(build_prompt_with_special_tokens(prompt, tokenizer), return_tensors='pt')
#     inputs_embeds = model.transformer.wte(inputs['input_ids'].to(device))
    
#     # Add linear transformation of student knowledge state with prompt tokens including delimiter ":" matching finetuning format
#     inputs_embeds = torch.add(inputs_embeds, linear(ks[0]))
    
#     # Generate student code by greedy decoding
#     outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs['attention_mask'].to(device))
#     generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#     return generated_code.strip(), code.strip(), prompt


# def compute_code_bleu(ground_truth_codes, generated_codes):
#     params='0.25,0.25,0.25,0.25'
#     lang='java'
#     codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    
#     return codebleu_score, detailed_codebleu_score
    

# class Metric():
#     """
#     Defines a text quality metric.
#     """
#     def get_name(self):
#         return self.name


#     @abc.abstractmethod
#     def compute_metric(self, texts):
#         pass


# class Distinct_N(Metric):

#     def __init__(self, n):
#         """
#         Distinct n-grams metrics. This is a sequence-level diversity metric.
#         See https://www.aclweb.org/anthology/N16-1014 for more details.

#         Args:
#             n (int): n-grams 
#         """
#         self.n = n
#         self.name = f'Distinct_{n}'


#     def compute_metric(self, texts):
#         return self._distinct_ngrams(texts, self.n)


#     def _distinct_ngrams(self, texts, n):
#         total = 0.0
#         for t in texts:
#             try:
#                 tokens = nltk.tokenize.word_tokenize(t)
#                 n_distinct = len(set(ngrams(tokens, n)))
#                 total += n_distinct/ len(tokens)
#             except Exception as e:
#                 print(f"Exception in computing Distinct_N metric: {e}")
#                 continue

#         return total / len(texts)


# @hydra.main(version_base=None, config_path=".", config_name="configs_okt")
# def main(configs):
#     # Make reproducible
#     set_random_seed(configs.seed)
#     # Set device
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
#     if configs.use_cuda: assert device.type == 'cuda', 'No GPU found'

#     train_set, valid_set, test_set, dataset, students = read_data(configs)
#     # Create a dictionary that maps student_id to index
#     student_id_to_index = {k: v for v, k in enumerate(students)}
#     tokenizer = create_tokenizer(configs)
#     collate_fn = CollateForOKT(tokenizer=tokenizer, configs=configs, student_id_to_index=student_id_to_index, device=device)
#     lstm_inputs = get_lstm_inputs(configs, tokenizer, student_id_to_index, train_set, dataset, collate_fn)

#     now = configs.checkpoint
#     configs.use_neptune = False
#     evaluate(configs, now, test_set, lstm_inputs, tokenizer, None, student_id_to_index, device)


# if __name__ == "__main__":
#     #torch.set_printoptions(profile="full")
#     main()