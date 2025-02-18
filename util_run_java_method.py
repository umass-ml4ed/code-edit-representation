import pandas as pd
import subprocess
import re
from jnius import autoclass
import sys
import threading
import signal 
import multiprocessing

from main_all_exp_history_bleu import *


# Function to generate codes for a batch of inputs
def generate_code_and_check_mask(model, history, dataset, tokenizer, configs, device):
    allowed_problemIDs = configs['allowed_problem_list']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    input_test_cases = pd.read_csv('data/input_test_cases.csv', index_col=False)
    history = serialize_dataset(history)
    dataset = serialize_dataset(dataset)
    print(len(dataset))
    return

    history_bleu = []
    personal_bleu = []
    model.eval()
    with torch.no_grad():
        for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating Model History", leave=False)):
            a1 = row['code_i']
            a1_mask = row['test_case_verdict_i']
            a2 = row['code_j']
            a2_mask = row['test_case_verdict_j']

            pid = row['problemID']
            sid = row['studentID']
            input_test_cases_for_this_pid = input_test_cases[input_test_cases['coding_prompt_id'] == int(pid)]
            zeroes = "0" * len(input_test_cases_for_this_pid)

            if a2_mask != a1_mask and a2_mask != zeroes:
                a1_emb = model.get_embeddings(a1)
                Da_hist, b2 = find_history_embedding(model=model, history=history, pid=pid, mask1=a1_mask, mask2=a2_mask)
                if Da_hist != None:
                    a2_gen = generate_code_from_vector(a1_emb + Da_hist, model, tokenizer, device)[0]
                    javafilename = get_java_file_name(pid, sid)
                    errcode = run_java_function(a2_gen, javafilename)
                    if errcode == None: #successful Compilation
                        mask_gen = get_test_case_mask(code=a2_gen, javafilename=javafilename, test_cases_for_this_pid=input_test_cases_for_this_pid)
                        # printCodePairSideBySide(a2, a2_gen, col_width=80)
                        print(a2_mask, mask_gen)
                        # bleu = compute_code_bleu([a2], [a2_gen])
                        # history_bleu.append(bleu)
                        # bleu = compute_code_bleu([b2], [a2_gen])
                        # personal_bleu.append(bleu)
                    if os.path.exists(javafilename+'.java'):
                        os.remove(javafilename+'.java')
                    if os.path.exists(javafilename+'.class'):
                        os.remove(javafilename+'.class')


            # if len(history_bleu) > 0: 
            #     print(f"History Bleu: {np.mean(history_bleu)}")
            #     print(f"Personal Bleu: {np.mean(personal_bleu)}")
            #     sys.stdout.flush()
                    # break

# Function to generate codes for a batch of inputs
def get_code_and_check_mask_gpt(dataset, tokenizer, configs, device):
    allowed_problemIDs = configs['allowed_problem_list']
    # print(len(dataset))
    # dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]
    # print(len(dataset))
    input_test_cases = pd.read_csv('data/input_test_cases.csv', index_col=False)
    # dataset = serialize_dataset(dataset)

    history_bleu = []
    personal_bleu = []
    for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating GPT Masks", leave=False)):
        a1 = row['code_i']
        a1_mask = row['test_case_verdict_i']
        a2 = row['code_j']
        a2_mask = row['test_case_verdict_j']

        pid = row['problemID']
        sid = row['studentID']
        input_test_cases_for_this_pid = input_test_cases[input_test_cases['coding_prompt_id'] == int(pid)]
        zeroes = "0" * len(input_test_cases_for_this_pid)

        if a2_mask != a1_mask and a2_mask != zeroes:
            a2_gen = row['code_gpt']
            javafilename = get_java_file_name(pid, sid)
            # printCodePairSideBySide(a2, a2_gen)
            errcode = run_java_function(a2_gen, javafilename)
            if errcode == None: #successful Compilation
                mask_gen = get_test_case_mask(code=a2_gen, javafilename=javafilename, test_cases_for_this_pid=input_test_cases_for_this_pid)
                # printCodePairSideBySide(a2, a2_gen, col_width=80)
                print(a2_mask, mask_gen)
            if os.path.exists(javafilename+'.java'):
                os.remove(javafilename+'.java')
            if os.path.exists(javafilename+'.class'):
                os.remove(javafilename+'.class')


def get_java_file_name(pid, sid):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    javafilename = 'test'+str(pid)+str(sid)+'_'+now
    return javafilename

def get_test_case_mask(code, javafilename, test_cases_for_this_pid):
    matches = ""
    for id, case in test_cases_for_this_pid.iterrows():
        input = case['input']
        try:
            output = execute_method_with_input_with_timeout(javafilename, code, input, 5)
        except Exception as e:
            output = None
        #outputs.append(output)
        expected_output = convert_to_datatype(case['expected_output'])
        #ex_outpputs.append(expected_output)
        #print(output, expected_output)
        if output == expected_output:
            matches = matches + '0'
        elif output == 'RTE':
            matches = matches + '2'
        elif output == 'TLE':
            matches = matches + '3'
        else: 
            matches = matches + '1'
    return matches

def convert_to_datatype(value):
    # Convert value to the appropriate data type
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        # Try converting to int
        return int(value)
    except ValueError:
        try:
            # Try converting to float
            return float(value)
        except ValueError:
            # If unable to convert to int or float, return as string
            return value[1:len(value)-1]

def parse_input_string(input_string):
    # Split the input string into individual values
    input_values = input_string.split(',')

    # Convert each value to the appropriate data type
    converted_values = []
    not_list = 1
    # converted_values = [convert_to_datatype(value.strip()) for value in input_values]
    for value in input_values:
        if value.strip() == 'new int[]{':
            not_list = 0
            tmp_list = []
            continue
        tmp_val = convert_to_datatype(value.strip())
        if not_list == 1:
            converted_values.append(tmp_val)
        elif value == '}':
            converted_values.append(tmp_list)
            not_list = 1
        else:
            tmp_list.append(tmp_val)
    return converted_values

def extract_method_names(java_code):
    method_names = []
    # Regular expression to match method declarations
    method_pattern = r'\b(?:public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])'
    matches = re.finditer(method_pattern, java_code)
    for match in matches:
        method_names.append(match.group(1))
    return method_names[0]

def get_method_signautre(code):
    lines = code.split('\n')
    return extract_method_names(lines[0])

def run_java_function(code, filename):
    with open(filename+'.java','w') as f:
        f.write('public class ' + filename + ' {\n')
        f.write(code)
        f.write('}')
    # 'javafiles/'+
    java_file = filename + '.java'
    # Compile Java code
    compile_process = subprocess.run(['javac', java_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if compile_process.returncode != 0:
        # print("Compilation failed:")
        # print(compile_process.stderr.decode())
        return compile_process.returncode

def execute_method_with_input(filename, code, input):
    method_name = get_method_signautre(code)
    # run_java_function(code, filename)
    Test = autoclass(filename)
    test = Test()
    method = getattr(test, method_name)
    args = parse_input_string(input)
    try:
        result = method(*args)
    except Exception as e:
        result = None
        result = "RTE"
    return result

def execute_method_with_input_with_timeout(filename, code, input, timeout=5):
    process = multiprocessing.Process(target=execute_method_with_input, args=(filename, code, input))
    process.start()
    process.join(timeout)

    # Check if the process is still alive (i.e., it timed out)
    if process.is_alive():
        # If it's still alive, terminate the process and raise a TimeoutError
        process.kill()
        process.join()  # Ensure the process terminates gracefully
        return "TLE"
        #raise TimeoutError("Method execution timed out")
    else:
        # If it's not alive, return the result of the method execution
        return execute_method_with_input(filename, code, input)


@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Current Device: ' + str(device))

    # Initialize the model
    tokenizer = create_tokenizer(configs)
    checkpoint_path = configs.model_save_dir

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs) #to keep the data constant over experiments
    
    # Path to the checkpoint
    # checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base

    # checkpoint_name = '20250130_211733' #cerdd, all, reconstruction =.5
    # checkpoint_name = '20250130_212046' #cerdd, all, reconstruction = 1
    # checkpoint_name = '20250130_212102' #cerdd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_212215' #cerdd, all, reconstruction = 2
    # checkpoint_name = '20250130_212223' #cerdd, all, reconstruction = 3

    # checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    # checkpoint_name = '20250130_212343' #cerd, all, reconstruction = 1
    # checkpoint_name = '20250130_213807' #cerd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2
    # checkpoint_name = '20250130_220007' #cerd, all, reconstruction = 3
    # checkpoint_name = '20250208_162240' #cerd, all, reconstruction = 4
    # checkpoint_name = '20250208_162301' #cerd, all, reconstruction = 5

    # checkpoint_name = '20250206_190729' #cerd, all, reconstruction = 3, regularization = 2

    # checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_212856' #cerd, all, reconstruction = 3, codet5-large
    # checkpoint_name = '20250211_212656' #cerdd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_213144' #cerdd, all, reconstruction = 3, codet5-large

    # checkpoint_name = '20250215_160225' #cerd, all, reconstruction = 2, codet5-base, rec and trans
    # checkpoint_name = '20250215_160712' #cerdd, all, reconstruction = 2, codet5-base, rec and trans

    # checkpoint_name = '20250216_021903' #cerd, all, recstruction = 1, contrastive = 1, regularization = 0
    # checkpoint_name = '20250216_022008' #cerd, all, recstruction = 0, contrastive = 1, regularization = 1
    # checkpoint_name = '20250216_022136' #cerd, all, recstruction = 1, contrastive = 0, regularization = 1

    # gptdf = pd.read_csv('baselines/gpt4o-code-test-case.csv')
    # get_code_and_check_mask_gpt(dataset=gptdf, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2, base
    print("checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2, base")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_and_check_mask(model= cerd_model, history=train_set, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)


# if __name__ == '__main__':    
#     targetpid = int(sys.argv[2])
#     dataset_filename = sys.argv[1]
#     #dataset_filename = 'tinydataset.pkl'
#     #dataset_filename = 'shortdataset.pkl'
#     #dataset_filename = 'largedataset.pkl'
#     file = pd.read_pickle(dataset_filename)
#     print('File loaded')
#     sys.stdout.flush()

#     test_cases = pd.read_csv('input_test_cases.csv')
    
#     okay_count = 0
#     total_sub = 0
#     progress = 0

#     print('Target Problem ' + str(targetpid))
#     sys.stdout.flush()
#     file = file[file['ProblemID'] == targetpid]

#     for index, row in file.iterrows():
#         progress = progress + 1
#         #print(progress)
#         #sys.stdout.flush()
#         code = row['Code']
#         score = row['Score_x']
#         pid = row['ProblemID']
#         sid = row['SubjectID']
#         cid = row['CodeStateID']
#         filename = 'test'+str(pid)+str(sid)+str(cid)

#         if pid != targetpid:
#             continue

#         test_cases_for_this_pid = test_cases[test_cases['coding_prompt_id'] == pid]
#         if len(test_cases_for_this_pid) == 0:
#             continue
#         errcode = run_java_function(code, filename)
#         if errcode == None:#successful compilation
#             total_sub = total_sub + 1
#             match = 0
#             total = 0
#             #outputs = []
#             #ex_outpputs = []
#             #inputs = []
#             matches = ""
#             for id, case in test_cases_for_this_pid.iterrows():
#                 input = case['input']
#                 #inputs.append(input)
#                 #print('Input')
#                 #print(input)
#                 try:
#                     output = execute_method_with_input_with_timeout(filename, code, input, 20)
#                 except Exception as e:
#                     output = None
#                 #outputs.append(output)
#                 expected_output = convert_to_datatype(case['expected_output'])
#                 #ex_outpputs.append(expected_output)
#                 #print(output, expected_output)
#                 if output == expected_output:
#                     match = match + 1
#                     matches = matches + '0'
#                 elif output == 'RTE':
#                     matches = matches + '2'
#                 elif output == 'TLE':
#                     matches = matches + '3'
#                 else: 
#                     matches = matches + '1'

#                 total = total + 1
#             score_calc = match*1.0 / total 
#             # print(outputs, ex_outpputs)
#             # print(match, total)
#             if abs(score - score_calc) < 1e-4:
#                 okay_count = okay_count + 1
#             #else:
#             #    print(pid)
#             #    print(code)
#             #    print(inputs)
#             #    print(outputs)
#             #    print(ex_outpputs)
#             print(pid, sid, matches, cid, score, score_calc, okay_count, total_sub)
#             sys.stdout.flush()
#         elif score > 0:#mistakenly compilation error
#             print('mistakenly compilation error')
#             sys.stdout.flush()

if __name__ == "__main__":
    main()            

