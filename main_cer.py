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
from eval import *
# from sentence_transformers import losses

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_latent_states(dataset, model):
    model.eval()
    res = np.empty((0,768))
    labels = []
    with torch.no_grad():
        for index, row in dataset.iterrows():
            A1 = row['code_i_1']
            A2 = row['code_j_1']
            B1 = row['code_i_2']
            B2 = row['code_j_2']
            
            Da, Db = model([A1, A2, B1, B2])
            Da = Da.cpu().detach().numpy()
            Db = Db.cpu().detach().numpy()
            res = np.concatenate((res, Da), axis=0)
            res = np.concatenate((res, Db), axis=0)
            labels.append(int(row['problemID']))
            labels.append(int(row['problemID']))
    return res, labels

def plot_clusters(dataloader, model, epoch, plotname, run):
    # X represents your high-dimensional data
        X, labels = get_latent_states(dataloader, model)
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)
        plt.close()
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)  # 'labels' should be your true or predicted labels
        plt.title('Epoch ' + str(epoch))
        plt.savefig(plotname + '.png')
        run['clusters/'+plotname].upload(plotname + '.png')

@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)
    
    # Sanity checks on config
    # sanitize_configs(configs)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    print('Current Device: ' + str(device))
    print('Loss Function: ' + str(configs.loss_fn))

    # Test on smaller fraction of dataset
    if configs.testing:
        # configs.use_neptune = False
        configs.epochs = 3
        # configs.save_model = False
    
    # Use neptune.ai to track experiments
    run = None
    if configs.use_neptune:
        if 'NEPTUNE_API_TOKEN' not in os.environ:
            print("Please set the NEPTUNE_API_TOKEN environment variable")
            sys.exit(1)
        # print(os.environ['NEPTUNE_API_TOKEN'])
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
    train_set, valid_set, test_set = read_data(configs)

    ## save the dataset along with the model
    if configs.save_model:
        # torch.save(dataset, os.path.join(configs.model_save_dir, now, 'dataset'))
        torch.save(train_set, os.path.join(configs.model_save_dir, now, 'train_set'))
        torch.save(valid_set, os.path.join(configs.model_save_dir, now, 'valid_set'))
        torch.save(test_set, os.path.join(configs.model_save_dir, now, 'test_set'))

    
    model,tokenizer = create_cer_model(configs, device)

    ## load data
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    train_loader = make_dataloader(train_set, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(valid_set, collate_fn=collate_fn, configs=configs)
    test_loader  = make_dataloader(test_set , collate_fn=collate_fn, configs=configs)

    # using different learning rates for different parts of the model
    # optimizer = optim.Adam([
    #                             {'params': model.pretrained_encoder.parameters(),   'lr': configs.lr_pretrained_encoder},
    #                             {'params': model.fc_edit_encoder.parameters(),      'lr': configs.lr_fc_edit_encoder},
    #                         ])
    
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    if configs.verbose == True: print(optimizer)

    # LR scheduler
    # num_training_steps = len(train_loader) * configs.epochs
    # num_warmup_steps = configs.warmup_ratio * num_training_steps
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    ## start training
    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')}
    
    if configs.loss_fn == 'ContrastiveLoss':
        criterion = ContrastiveLoss(device=device, margin=configs.margin)
        # criterion = losses.ContrastiveLoss(model=None, margin=configs.margin)
    elif configs.loss_fn == 'CosineSimilarityLoss':
        criterion = CosineSimilarityLoss(device=device)
    elif configs.loss_fn == 'NTXentLoss' : # not relevant right now
        criterion = NTXentLoss(device=device, batch_size=configs.batch_size)
    elif configs.loss_fn == 'MultipleNegativesRankingLoss':
        criterion = MultipleNegativesRankingLoss(device=device)

    print(len(train_set))
    
    for ep in tqdm(range(configs.epochs), desc="epochs"):
        sys.stdout.flush()  # Manually flush the buffer
        model.train()
        train_logs, test_logs, valid_logs = [], [], []
        
        ## training
        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            train_log = training_step(batch, idx, len(train_loader), model, criterion, optimizer, configs, device=device)
            train_logs.append(train_log)

            if configs.verbose == True and configs.show_loss_at_every_epoch == True:
                print("Epoch: " + str(ep) + " Train Loss: " + str (train_log['loss']))
                # print(train_log["output1"])
        

        ## validation
        # if valid_set:
        for idx, batch in enumerate(tqdm(valid_loader, desc="validation", leave=False)):
            valid_log = training_step(batch, idx, len(valid_loader), model, criterion, optimizer, configs, device=device)
            valid_logs.append(valid_log)
            
        ## testing
        # if test_set:
        for idx, batch in enumerate(tqdm(test_loader, desc="testing", leave=False)):
            test_log = training_step(batch, idx, len(test_loader), model, criterion, optimizer, configs, device=device)
            test_logs.append(test_log)

        if ep % 1 == 0:
            if configs.show_accuracy_at_every_epoch == True or configs.use_neptune == True:
                train_accuracy, test_accuracy, valid_accuracy = get_model_accuracy(configs, model, train_set, test_set, valid_set)
            
            if configs.verbose == True and configs.show_accuracy_at_every_epoch == True:
                print("Train Accuracy: ", train_accuracy)
                print("Test Accuracy: ", test_accuracy)
                print("Valid Accuracy: ", valid_accuracy)
        
        ## logging
        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs )
        
        ## log the results and save models
        # if valid_set != None:
        for key in valid_logs:
            if key == 'loss':
                if( float(valid_logs[key]) < best_valid_metrics[key] ):
                    best_valid_metrics[key] = float(valid_logs[key])
                    best_metrics_with_valid[key] = float(test_logs[key])
                    ## Save the model with lowest validation loss
                    if configs.save_model:
                        if configs.use_neptune:
                            run["best_model_at_epoch"].log(ep)
                        torch.save(model, os.path.join(configs.model_save_dir, now, 'model'))
                    
                    plot_clusters(train_set, model, ep, 'train_cluster', run)
                    plot_clusters(valid_set, model, ep, 'valid_cluster', run)
                    plot_clusters(test_set, model, ep, 'test_cluster', run)
                            
        # if test_set != None:
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
            run["metrics/train_accuracy"].log(train_accuracy)
            run["metrics/test_accuracy"].log(test_accuracy)
            run["metrics/valid_accuracy"].log(valid_accuracy)

    # Evaluation post training for accuracy
    train_accuracy, test_accuracy, valid_accuracy = get_model_accuracy(configs, model, train_set, test_set, valid_set)
    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)
    print("Valid Accuracy: ", valid_accuracy)
    # plot_clusters(train_set, model, ep, 'train_cluster', run)
    # plot_clusters(valid_set, model, ep, 'valid_cluster', run)
    # plot_clusters(test_set, model, ep, 'test_cluster', run)
        
    if configs.use_neptune:
        run["metrics/train_accuracy"].log(train_accuracy)
        run["metrics/test_accuracy"].log(test_accuracy)
        run["metrics/valid_accuracy"].log(valid_accuracy)



if __name__ == "__main__":
    #torch.set_printoptions(profile="full")
    main()
