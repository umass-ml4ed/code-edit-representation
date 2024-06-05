import os
from datetime import datetime
import torch.optim as optim
import transformers
import neptune
from neptune.utils import stringify_unsupported
import hydra
from omegaconf import OmegaConf
import sys

from data_loader import *
from model import *
from trainer import *
from utils import *
# from eval import *


# def sanitize_configs(configs):
    # assert ( not (configs.use_lstm and configs.use_h_bar_static) ), "Enable either LSTM for time varying h bar or enable static h bar"
    # assert ( (configs.use_lstm == False and configs.lstm_hid_dim == 0) or (configs.use_lstm == True and configs.lstm_hid_dim > 0) ), "Invalid combination of configs use_lstm and lstm_hid_dim"
    # assert ( (configs.use_h_bar_static == False and configs.h_bar_static_dim == 0) or (configs.use_h_bar_static == True and configs.h_bar_static_dim > 0) ), "Invalid combination of configs use_h_bar_static and h_bar_static_dim"
    # assert ( (configs.use_q_model == False and ((configs.dim_normal + configs.dim_categorical) == 0)) or (configs.use_q_model == True and ((configs.dim_normal + configs.dim_categorical) > 0)) ), "Invalid combination of configs use_q_model and dim_normal and dim_categorical"


@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanity checks on config
    # sanitize_configs(configs)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    # if configs.use_cuda: 
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     # assert device.type == 'cuda', 'No GPU found'
    # # Apple metal acceleration: don't enable for now since some operations are not implemented in MPS and torch.gather has an issue (https://github.com/pytorch/pytorch/issues/94765)
    # #elif( torch.backends.mps.is_available() ):
    # #    device = torch.device("mps")
    # else:
    #     device = torch.device('cpu')    
    
    print(device)

    # Test on smaller fraction of dataset
    if configs.testing:
        configs.use_neptune = False
        configs.epochs = 2
        configs.save_model = False
    
    # Use neptune.ai to track experiments
    run = None
    if configs.use_neptune:
        if 'NEPTUNE_API_TOKEN' not in os.environ:
            print("Please set the NEPTUNE_API_TOKEN environment variable")
            sys.exit(1)
        print(os.environ['NEPTUNE_API_TOKEN'])
        run = neptune.init_run(
            project=configs.neptune_project,
            
            #Set Neptune API token as an environment variable. This is a secure way to pass the token. The token can also be saved in the config directly, but it must be removed before pushing to git.
            api_token=os.environ['NEPTUNE_API_TOKEN'], 
            capture_hardware_metrics = False,
            name=configs.exp_name + '_{}'.format(now), # mark the experiment using the current date and time
            custom_run_id=configs.exp_name + '_{}'.format(now),
            tags=[now],
        )
        run["parameters"] = stringify_unsupported(OmegaConf.to_container(configs, resolve=True))
        run['time'] = now

    if configs.save_model:
        os.makedirs(os.path.join(configs.model_save_dir, now))

    ## load the init dataset
    train_set, valid_set, test_set, dataset = read_data(configs)

    ## save the dataset along with the model
    if configs.save_model:
        torch.save(dataset, os.path.join(configs.model_save_dir, now, 'dataset'))
        torch.save(train_set, os.path.join(configs.model_save_dir, now, 'train_set'))
        torch.save(valid_set, os.path.join(configs.model_save_dir, now, 'valid_set'))
        torch.save(test_set, os.path.join(configs.model_save_dir, now, 'test_set'))

    # # Create a dictionary that maps student_id to index
    # student_id_to_index = {k: v for v, k in enumerate(students)}

    ## load model
    # lstm, tokenizer, model, linear, q_model, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, student_params_h_hat_discrete, student_params_h_hat_discrete_copy = create_okt_model(configs, students, device) 

    model,tokenizer = create_cer_model(configs, device)

    ## load data
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    train_loader = make_dataloader(train_set, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(valid_set, collate_fn=collate_fn, configs=configs)
    test_loader  = make_dataloader(test_set , collate_fn=collate_fn, configs=configs)

    # ## optimizers and loss function
    # optimizers_generator = []
    # optimizer_lm = transformers.AdamW(model.parameters(), lr=configs.lr, correct_bias=True)
    # optimizers_generator.append(optimizer_lm)
    # optimizer_linear = optim.Adam(linear.parameters(), lr=configs.lr_linear)
    # optimizers_generator.append(optimizer_linear)

    # ## optimizer for lstm
    # optimizers_lstm = None
    # if configs.train_lstm and configs.use_lstm:
    #     optimizers_lstm = []
    #     optimizer_lstm = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
    #     optimizers_lstm.append(optimizer_lstm)
        
    # ## optimizer for q model
    # optimizers_q = None
    # if configs.use_q_model:
    #     optimizers_q = []
    #     optimizer_q = optim.Adam(q_model.parameters(), lr=configs.lr_q)
    #     optimizers_q.append(optimizer_q)
    #     if( configs.use_h_bar_static ):
    #         optimizer_student_params_h_bar_static = optim.Adam([student_params_h_bar_static], lr=configs.lr_q)
    #         optimizers_q.append(optimizer_student_params_h_bar_static)
    #     if( configs.dim_normal > 0 ):
    #         optimizer_student_params_h_hat_mu = optim.Adam([student_params_h_hat_mu], lr=configs.lr_q)
    #         optimizers_q.append(optimizer_student_params_h_hat_mu)
    #         if( configs.learn_sigma ):
    #             optimizer_student_params_h_hat_sigma = optim.Adam([student_params_h_hat_sigma], lr=configs.lr_q)
    #             optimizers_q.append(optimizer_student_params_h_hat_sigma)
    #     if( configs.dim_categorical > 0 ):
    #         optimizer_student_params_h_hat_discrete = optim.Adam([student_params_h_hat_discrete], lr=configs.lr_q)
    #         optimizers_q.append(optimizer_student_params_h_hat_discrete)
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)

    # LR scheduler
    num_training_steps = len(train_loader) * configs.epochs
    num_warmup_steps = configs.warmup_ratio * num_training_steps
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    ## start training
    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')}
    
    criterion = ContrastiveLoss(device=device)
    model.train()
    # for ep in tqdm(range(configs.epochs), desc="epochs"):
    #     for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
    #         A1 = batch['A1']
    #         A2 = batch['A2']
    #         B1 = batch['B1']
    #         B2 = batch['B2']
    #         label = batch['label']

    #         optimizer.zero_grad()
            
    #         outputs = model(A1, A2, B1, B2)
    #         loss = criterion(outputs, label)
    #         loss.backward()
    #         optimizer.step()

    for ep in tqdm(range(configs.epochs), desc="epochs"):
        train_logs, test_logs, valid_logs = [], [], []
        
        ## training
        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            train_log = generator_step(batch, model, criterion, optimizer, scheduler, configs, device=device)
            train_logs.append(train_log)
            ## save results to neptune.ai
            if configs.log_train_every_itr and configs.use_neptune:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs)
                    for key in itr_train_logs:
                        run["metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key)].log(itr_train_logs[key])

        ## validation
        for idx, batch in enumerate(tqdm(valid_loader, desc="validation", leave=False)):
            valid_log = generator_step(batch, model, criterion, optimizer, scheduler, configs, device=device)
            valid_logs.append(valid_log)
            
        ## testing
        for idx, batch in enumerate(tqdm(test_loader, desc="testing", leave=False)):
            test_log = generator_step(batch, model, criterion, optimizer, scheduler, configs, device=device)
            test_logs.append(test_log)
        
        ## logging
        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs )
        
        ## log the results and save models
        for key in valid_logs:
            ## only one key (loss) available for OKT
            if key == 'loss':
                if( float(valid_logs[key]) < best_valid_metrics[key] ):
                    best_valid_metrics[key] = float(valid_logs[key])
                    best_metrics_with_valid[key] = float(test_logs[key])
                    ## Save the model with lowest validation loss
                    if configs.save_model:
                        if configs.use_neptune:
                            run["best_model_at_epoch"].log(ep)
                        torch.save(model, os.path.join(configs.model_save_dir, now, 'model'))
                            
        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])

        ## save results to neptune.ai
        if configs.use_neptune:
            for key in train_logs:
                run["metrics/train/"+key].log(train_logs[key])
            for key in valid_logs:
                run["metrics/valid/"+key].log(valid_logs[key])
            for key in test_logs:
                run["metrics/test/"+key].log(test_logs[key])
            for key in best_valid_metrics:
                run["metrics/valid/best_"+key].log(best_valid_metrics[key])
            for key in best_test_metrics:
                run["metrics/test/best_"+key].log(best_test_metrics[key])
            for key in best_metrics_with_valid:
                run["metrics/test/best_"+key+"_with_valid"].log(best_metrics_with_valid[key])
            run["epoch"].log(ep)
    
    # # Evaluation post training for code generation on test set and CodeBleu
    # evaluate(configs, now, test_set, lstm_inputs, tokenizer, run, student_id_to_index, device)


if __name__ == "__main__":
    #torch.set_printoptions(profile="full")
    main()
