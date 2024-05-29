import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # Assuming label is 1 for similar pairs and 0 for dissimilar pairs
        loss = 0.5 * (label * output**2 + (1 - label) * F.relu(self.margin - output).pow(2))
        return loss.mean()



def normal_nll(mu, sigma, x):
    nll =  torch.sum( torch.log(sigma) ) + torch.mul( torch.div( x.size(0), 2.0 ), torch.log( torch.mul(2, np.pi) ) ) + torch.sum( torch.div( torch.square( torch.sub(x, mu) ), torch.mul( 2, torch.square(sigma) ) ) )
    
    return nll

def generator_step(batch, model, criterion, optimizer, scheduler, configs, device):
    A1 = batch['A1']
    A2 = batch['A2']
    B1 = batch['B1']
    B2 = batch['B2']
    label = batch['label']

    optimizer.zero_grad()
    
    outputs = model(A1, A2, B1, B2)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    log = {'loss': loss.cpu().detach()}
    return log

# def generator_step(batch, lstm_inputs, model, lstm, linear, q_model, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, 
#                    student_params_h_hat_discrete, optimizers=None, optimizers_lstm=None, optimizers_q=None,
#                    configs=None, train=True, scheduler=None, device=None, student_params_h_hat_discrete_copy=None):    
#     if train:
#         if configs.train_okt:
#             assert(optimizers != None)
#             model.train()
#             linear.train()
#         if configs.train_lstm and configs.use_lstm:
#             assert(optimizers_lstm != None)
#             lstm.train()
#         if configs.use_q_model:
#             assert(optimizers_q != None)
#             q_model.train()
#     else:
#         model.eval()
#         linear.eval()
#         if configs.use_lstm:
#             lstm.eval()
#         if configs.use_q_model:
#             q_model.eval()
        
#     # assemble generator input
#     generator_inputs_ids, attention_mask, labels, prompt_id_lens, students, timesteps, student_ids = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
    
#     generator_inputs_wte, ks, h_hat_cont, h_hat_disc = assemble_generator_input(model, lstm, linear, student_params_h_bar_static, student_params_h_hat_mu, 
#                                                                                 student_params_h_hat_sigma, student_params_h_hat_discrete, configs,
#                                                                                 generator_inputs_ids, prompt_id_lens, lstm_inputs, students, timesteps, 
#                                                                                 student_ids, device, generation=False)
    
#     # forward generator
#     if train:
#         outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
#     else:
#         with torch.no_grad():
#             outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
    
#     # compute stats
#     loss_okt = outputs["loss"]
    

#     # Compute Q loss
#     loss_q_discrete = torch.tensor([0.0], requires_grad=True).to(device)
#     loss_q_normal = torch.tensor([0.0], requires_grad=True).to(device)
#     loss_q = torch.tensor([0.0], requires_grad=True).to(device)
#     # We don't add Q loss when h hat is not sampled during training for a performance comparison with equivalent non stochastic model
#     if( configs.use_q_model and configs.sample_h_hat_train ):
#         hidden_states = outputs['hidden_states'][-1] # Shape = [B, T, D]
#         # Compute mean hidden state corresponding to labels != -100 (code tokens only)
#         mask = (labels != -100).float() # Shape = [B, T]
#         numerator = torch.sum( torch.mul(hidden_states, mask.unsqueeze(-1)), dim=1 ) # Shape = [B, D]
#         denominator = torch.sum( mask.unsqueeze(-1), dim=1 ) # Shape = [B, 1]
#         mean_hidden_states = torch.div(numerator, denominator) # Shape = [B, D]

#         mu_pred, sigma_pred, discrete_logits_pred = q_model(mean_hidden_states) # Shapes = [B, D_cont], [B, D_cont], [B, (D_disc * Num_classes_disc)]

#         # Compute loss for continuous normal latent h hat
#         if( configs.dim_normal > 0 ):
#             # Compute negative log likelihood of h_hat_cont under normal parameterized by mu_pred and sigma_pred
#             batch_size = h_hat_cont.shape[0]
#             loss_q_normal = normal_nll(torch.flatten(mu_pred), torch.flatten(sigma_pred), torch.flatten(h_hat_cont)) / (float(batch_size) * configs.dim_normal)
#         # Compute loss for discrete categorical latent h hat
#         if( configs.dim_categorical > 0 ):
#             loss_fct = nn.CrossEntropyLoss()
#             # Reduction is mean by default: averaged by product of batch size and number of dims 
#             loss_q_discrete = loss_fct(discrete_logits_pred.view(-1, configs.num_classes_categorical), h_hat_disc.view(-1, configs.num_classes_categorical))
#         loss_q = (configs.weight_q_loss_continuous * loss_q_normal) + (configs.weight_q_loss_discrete * loss_q_discrete)
    
#     if( configs.compare_without_info_reg ):
#         # For comparison of Q model performance without information regularization (Q loss) added in loss
#         loss = loss_okt
#     else:
#         loss = loss_okt + loss_q
    
#     if train:
#         loss.backward()
    
#     # optimization
#     if train:
#         # Training the LM and linear layer for ks alignment with problem token embeddings
#         for optimizer in optimizers:
#             optimizer.step()
#         if configs.use_scheduler:
#             scheduler.step()
#         for optimizer in optimizers:
#             optimizer.zero_grad()
        
#         # training the lstm
#         if configs.train_lstm and configs.use_lstm:
#             assert(optimizers_lstm != None)
#             for optimizer in optimizers_lstm:
#                 optimizer.step()
#             for optimizer in optimizers_lstm:
#                 optimizer.zero_grad()
        
#         # training the q model
#         if configs.use_q_model:
#             assert(optimizers_q != None)
#             for optimizer in optimizers_q:
#                 optimizer.step()
#             for optimizer in optimizers_q:
#                 optimizer.zero_grad()
    
#     log = {'loss': loss.cpu().detach(),
#            'loss_okt': outputs["loss"].cpu().detach()}
#     if( configs.use_q_model ):
#         if( configs.sample_h_hat_train ):
#             log['loss_q'] = loss_q.cpu().detach()
#             log['loss_q_normal'] = loss_q_normal.cpu().detach()
#             log['loss_q_discrete'] = loss_q_discrete.cpu().detach()
#         if( train and configs.dim_categorical > 0 ):
#             log['student_params_h_hat_discrete mean change'] = torch.mean(torch.abs(student_params_h_hat_discrete_copy - student_params_h_hat_discrete.data)).cpu().detach()
#             cat_dist = torch.distributions.Categorical(logits=student_params_h_hat_discrete)
#             cat_dist_entropy = cat_dist.entropy()
#             log['student_params_h_hat_discrete entropy'] = torch.mean(cat_dist_entropy).cpu().detach()

#     return log
    

def get_knowledge_states_for_generator(lstm, lstm_inputs, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, 
                                       student_params_h_hat_discrete, students, timesteps, configs, student_ids, device, generation=False):
    '''
    used during ***inference (generation) time*** to get a student's knowledge state
    '''
    ks = None
    h_hat_cont = None
    h_hat_disc = None

    if configs.use_lstm:
        # get lstm inputs
        lstm_ins = [lstm_inputs[s] for s in students]
        # TODO p2: vectorize
        max_len = max(len(i) for i in lstm_ins)
        padded_lstm_ins = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in lstm_ins]
        padded_lstm_ins = torch.stack([torch.stack(x, dim=0) for x in padded_lstm_ins], dim=1).float() # Shape = [T, B, D_bar]
        # Get student knowledge states
        if( configs.train_lstm and not generation ):
            out, hidden = lstm(padded_lstm_ins.to(device)) # Shape = [T, B, D_bar]
        else:
            with torch.no_grad():
                out, hidden = lstm(padded_lstm_ins.to(device)) # Shape = [T, B, D_bar]
        ks = out[timesteps, list(range(out.shape[1])), :] # Extract the hidden states -> shape = [B, D_bar]
    
    # Enable either LSTM for time varying h bar or enable static h bar
    elif( configs.use_h_bar_static ):
        index = student_ids.unsqueeze(-1).repeat(1, configs.h_bar_static_dim) # Shape = [B, D_bar]
        h_bar = torch.gather(student_params_h_bar_static, dim=0, index=index) # Shape = [B, D_bar]
        ks = h_bar # Shape = [B, D_bar]
    
    if( configs.use_q_model ):
        if( configs.dim_normal > 0 ):
            # Gather student parameters across rows (dim=0)
            index = student_ids.unsqueeze(-1).repeat(1, configs.dim_normal) # Shape = [B, D_cont]
            mu = torch.gather(student_params_h_hat_mu, dim=0, index=index) # Shape = [B, D_cont]
            sigma = torch.gather(student_params_h_hat_sigma, dim=0, index=index) # Shape = [B, D_cont]
            if( generation ):
                # Use mu as h hat during code generation, no sampling performed
                h_hat_cont = mu # [B, D_cont]
            else:
                if( configs.sample_h_hat_train ):
                    # Create a multivariate normal distribution using flattened mu and sigma for efficiency
                    # MultivariateNormal expects a covariance matrix unlike Normal which expects the scale (std dev), therefore square the std dev for input
                    # Exponentiate since log sigma is stored as parameters
                    mvn = torch.distributions.multivariate_normal.MultivariateNormal(torch.flatten(mu), covariance_matrix = torch.diag(torch.pow(torch.exp(torch.flatten(sigma)), 2)))
                    # Sample normal latent h hat using reparameterization trick allowing for z
                    h_hat_cont = mvn.rsample() # [(B*D_cont)]
                    h_hat_cont = torch.reshape(h_hat_cont, (-1, configs.dim_normal)) # [B, D_cont]
                else:
                    # Don't sample h hat during training for a performance comparison with equivalent non stochastic model
                    h_hat_cont = mu # [B, D_cont]
            # Concatenate h hat cont with existing knowledge state (h bar from output of LSTM) if not None
            ks = torch.concat((ks, h_hat_cont), dim=-1) if ks is not None else h_hat_cont # Shape = [B, (D_bar + D_cont)]           
        
        if( configs.dim_categorical > 0 ):
            # Gather student parameters across rows (dim=0)
            index = student_ids.unsqueeze(-1).repeat(1, configs.dim_categorical).unsqueeze(-1).repeat(1, 1, configs.num_classes_categorical) # [B, D_disc, Num_classes_disc]
            discrete_logits = torch.gather(student_params_h_hat_discrete, dim=0, index=index) # Shape = [B, D_disc, Num_classes_disc]
            if( generation ):
                # No sampling performed during generation
                if( configs.sample_h_hat_train ):
                    # Use argmax as class index during code generation, keep one-hot format as in training when configs.sample_h_hat_train=True
                    h_hat_disc = F.one_hot(torch.argmax(discrete_logits, dim=-1), num_classes=configs.num_classes_categorical) # Shape = [B, D_disc, Num_classes_disc]
                else:
                    # Keep softmax format as in training when configs.sample_h_hat_train=False
                    h_hat_disc = F.softmax(discrete_logits, dim=-1) # Shape = [B, D_disc, Num_classes_disc]
            else:
                if( configs.sample_h_hat_train ):
                    # Sample h hat disc using Gumbel-Softmax trick to allow backpropagation in one-hot manner (hard=True)
                    h_hat_disc = F.gumbel_softmax(discrete_logits, tau=1, hard=True) # Shape = [B, D_disc, Num_classes_disc]
                else:
                    # Don't sample h hat disc during training for a performance comparison with equivalent non stochastic model, softmax instead of one-hot for differentiability
                    h_hat_disc = F.softmax(discrete_logits, dim=-1) # Shape = [B, D_disc, Num_classes_disc]
            h_hat_disc = torch.reshape(h_hat_disc, (-1, configs.dim_categorical * configs.num_classes_categorical)) # Shape = [B, (D_disc * Num_classes_disc)]
            # Concatenate h hat disc with existing knowledge state (h hat cont and h bar from output of LSTM) if not None
            ks = torch.concat((ks, h_hat_disc), dim=-1) if ks is not None else h_hat_disc # Shape = [B, (D_bar + D_cont + (D_disc * Num_classes_disc))]

    return ks, h_hat_cont, h_hat_disc


def assemble_generator_input(model, lstm, linear, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, student_params_h_hat_discrete, configs,
                             generator_input_ids, prompt_id_lens, lstm_inputs, students, timesteps, student_ids, device, generation=False):
    '''
    linear: linear transform the knowledge state before adding in with the generator input
    '''
    
    # compute generator embeddings for the batch
    generator_input_wte = model.transformer.wte(generator_input_ids) # Shape = [B, T, 768]
    
    # get knowledge states
    ks, h_hat_cont, h_hat_disc = get_knowledge_states_for_generator(lstm, lstm_inputs, student_params_h_bar_static, student_params_h_hat_mu, 
                                                                    student_params_h_hat_sigma, student_params_h_hat_discrete, students, timesteps, 
                                                                    configs, student_ids, device, generation)
    
    # Add linear transformation of student knowledge state with only prompt tokens

    ks = linear(ks) # Shape = [B, 768]
    ks = ks.unsqueeze(1).repeat(1, generator_input_wte.size(1), 1) # Shape = [B, T, 768]
    range_tensor = torch.arange(generator_input_ids.size(1), device=device).unsqueeze(0) # Shape = [1, T]
    range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1) # Shape = [B, T]
    mask_tensor = (range_tensor >= prompt_id_lens.unsqueeze(-1)) # Shape = [B, T]
    ks[mask_tensor] = torch.zeros(ks.size(-1), device=device) # Shape = [B, T, 768]
    generator_input_wte = torch.add(generator_input_wte, ks) # Shape = [B, T, 768]
        
    return generator_input_wte, ks, h_hat_cont, h_hat_disc