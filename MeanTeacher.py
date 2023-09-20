#!coding:utf-8
import torch
from torch.nn import functional as F
import random
import time
import os
import datetime
from pathlib import Path
from collections import defaultdict
from itertools import cycle
from utils.data_utils import Similarity_Matrix
from utils.data_utils import Pairwise_Matrix
from utils.LogLoss import PairLoss
from utils.ramps import exp_rampup
from utils.mixup import *

from utils.loss import mse_with_softmax
from utils.calc_map import calc_map
from utils.LogLoss import ConsistenceLoss
from utils.LogLoss import SupConLoss
from utils.LogLoss import image_wise_Loss

from utils.dist import cosine_dist
from utils.dist import eucl_dist

from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model, ema_model, optimizer, device, config):
        self.model      = model
        self.ema_model  = ema_model
        self.optimizer  = optimizer
        self.lce_loss   = torch.nn.CrossEntropyLoss()
        self.uce_loss   = torch.nn.CrossEntropyLoss(reduction='none')
        self.save_dir   = '{}-{}_{}'.format(
                          config.dataset, config.num_labels,
                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_dir    = os.path.join(config.save_dir, self.save_dir)
        self.usp_weight  = config.usp_weight
        self.cons_weight = config.cons_weight
        self.contras_weight = config.contras_weight

        self.threshold   = config.threshold
        self.ema_decay   = config.ema_decay
        self.rampup      = exp_rampup(config.weight_rampup) #30
        self.contrastive_up = exp_rampup(30)
        self.save_freq   = config.save_freq
        self.print_freq  = config.print_freq
        self.device      = device
        self.global_step = 0
        self.epoch       = 0

        self.pair_loss = PairLoss()
        self.code_bits = config.code_bits
        self.num_classes = config.num_classes

        self.con_loss = ConsistenceLoss(temperature=config.consis_t)
        self.contrastive_loss = SupConLoss(temperature=config.contras_t1)
        self.image_loss = image_wise_Loss(temperature=config.contras_t2, device=self.device)
        
        self.queue_idx = 0
        self.queue_idx_un = 0
        self.writer = SummaryWriter()

    def train_iteration(self, label_loader, unlab_loader, print_freq):
        epoch_start = time.time()  # start time
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0

        for (x1, label_y, ldx), ((uw, us), unlab_y) in zip(cycle(label_loader), unlab_loader):
            self.global_step += 1; batch_idx += 1;
            label_x, unlab_uw, unlab_us = x1.to(self.device), uw.to(self.device), us.to(self.device)
            label_y, unlab_y = label_y.to(self.device), unlab_y.to(self.device)
            ##=== decode targets ===

            lbs, ubs = x1.size(0), uw.size(0)

            ##=== forward ===
            features, outputs, code = self.model(label_x)
            # supervised_classification loss
            loss = self.lce_loss(outputs, label_y)
            loop_info['lloss'].append(loss.item())

            # supervised_hashing loss
            S = Similarity_Matrix(label_y)
            S = S.to(self.device)
            pairloss = self.pair_loss(S, code)
            loss = loss + pairloss
            loop_info['lhash'].append(pairloss.item())

            ##=== Semi-supervised Training ===
            ## update mean-teacher
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
            ## use the outputs of weak unlabeled data as pseudo labels
            feature_us, outpot_us, code_us = self.model(unlab_us)
            feature_uw, _, code_usw = self.model(unlab_uw)

            with torch.no_grad():
                feat_uw, output_uw, code_uw = self.ema_model(unlab_uw)

                code_uw = code_uw.detach()
                feat_uw = feat_uw.detach()
                output_uw = output_uw.detach()
                ###
                #计算伪标签
                
                index_buffer = random.sample(range(0, 1000), 100)

                
                unlabel_label_dist = eucl_dist(output_uw, self.pred_buffer[index_buffer])
                First_top = torch.argmin(unlabel_label_dist, dim=1)
                Clone_data_label = self.label_buffer[index_buffer][First_top]
                matrix_mask = Pairwise_Matrix(Clone_data_label, self.label_buffer[index_buffer]).to(self.device)
                cos_dist = cosine_dist(output_uw, self.pred_buffer[index_buffer])
                cos_mask = cos_dist * matrix_mask
                
                                               
                if self.epoch > 5:


                    index_neiboard = random.sample(range(0, 1000), 500)

                    pairwise_dist = eucl_dist(output_uw, self.un_pred_buffer[index_neiboard])  # 欧式
                    A = torch.tensor(1.0) / (pairwise_dist + 1e-5)  # 1e-5
                    top_idx = torch.argsort(A, 1, descending=True)[:,13:]

                    A.scatter_(dim=1, index=top_idx, src=torch.zeros([100, 487]).to(self.device))
                    A = A / torch.sum(A, dim=1, keepdim=True)

                    alpha1 = 1.0
                    alpha2 = 1.0 - alpha1

                    alpha_a = torch.tensor(alpha1).to(self.device).detach()
                    alpha_b = torch.tensor(alpha2).to(self.device).detach()
                    code_uw = alpha_b * code_uw + alpha_a * torch.mm(A, self.un_code_buffer[index_neiboard])
                
                   
            cons_loss = self.contrastive_loss(code_us, self.code_buffer[index_buffer], positive_mask=matrix_mask, weight_mask=cos_mask)*self.cons_weight
            loss = loss + cons_loss * self.rampup(self.epoch)
            loop_info['ucontrstive'].append(cons_loss.item())
            
            
            contras_loss = self.image_loss(code_usw, code_us) * self.contras_weight
            loss = loss + contras_loss * self.contrastive_up(self.epoch)
            loop_info['uimage'].append(contras_loss.item())
            
            ##  consistency unsupervised hashing

                 
            conloss = self.con_loss(code_us, code_uw)*self.rampup(self.epoch)*self.usp_weight
            loss = loss + conloss
            loop_info['uhash'].append(conloss.item())
            self.writer.add_scalar('rampup', self.rampup(self.epoch), global_step=self.epoch)
            
            

            ##=== backwark ===
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##=== log info ===

            ##update memory bank
            with torch.no_grad():
                self.code_buffer[self.queue_idx: self.queue_idx+lbs] = code.clone().detach()
                self.label_buffer[self.queue_idx: self.queue_idx+lbs] = label_y
                self.pred_buffer[self.queue_idx: self.queue_idx+lbs] = outputs.clone().detach()
                self.queue_idx = (self.queue_idx + lbs) % 1000

                self.un_pred_buffer[self.queue_idx_un: self.queue_idx_un+lbs] = outputs.clone().detach()
                self.un_code_buffer[self.queue_idx_un: self.queue_idx_un+lbs] = code.clone().detach()
                self.queue_idx_un = (self.queue_idx_un+lbs) % 1000


            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(label_y.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['uacc'].append(unlab_y.eq(output_uw.max(1)[1]).float().sum().item())
            if print_freq>0 and (batch_idx % print_freq)==0:
                print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        loop_time = time.time() - epoch_start
        print(f">>>[Train Time]: {loop_time:.2f}s", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            ##=== forward ===
            _, outputs, _ = self.model(data)
            _, ema_outputs, _ = self.ema_model(data)

            ##=== log info ===
            label_n, unlab_n = label_n+lbs, unlab_n+ubs
            loop_info['lacc'].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            loop_info['l2acc'].append(targets.eq(ema_outputs.max(1)[1]).float().sum().item())
            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[test]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    # evaluate map
    def evaluate(self, query_loader, database_loader, code_length):
        loop_info = defaultdict(list)
        # Generate hash code
        query_code, query_code_ema, query_targets = self.generate_code(query_loader, code_length)
        database_code, database_code_ema, database_targets = self.generate_code(database_loader, code_length)
        # Compute map
        meanAP = calc_map(query_code, database_code, query_targets, database_targets, len(database_loader.dataset))
        meanAP_ema = calc_map(query_code_ema, database_code_ema, query_targets, database_targets, len(database_loader.dataset))
        ##=== log info ===

        loop_info['meanAP'].append(meanAP.item())
        loop_info['meanAP_ema'].append(meanAP_ema.item())
        print(">>>[Evaluate]", self.gen_info(loop_info, 1, 1))

        return loop_info

    # generate code
    def generate_code(self, data_loader, code_length):

        code = []
        code_ema = []
        targets = []
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(self.device)
            ##=== forward ===
            _, _, outputs = self.model(data)
            code.extend(outputs.sign().cpu().numpy())
            _, _, outputs_ema = self.ema_model(data)
            code_ema.extend(outputs_ema.sign().cpu().numpy())
            targets.extend(label)
        code = torch.tensor(np.array(code)).to(self.device)
        code_ema = torch.tensor(np.array(code_ema)).to(self.device)

        return code, code_ema, np.array(targets)

    def train(self, label_loader, unlab_loader, print_freq=20):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, print_freq)

    def eval(self, query_loader, database_loader, code_length):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.evaluate(query_loader, database_loader, code_length)

    def loop(self, epochs, label_data, unlab_data, query_data, database, scheduler=None):
        best_acc, n, best_info, best_map, best_map_ema, map_info, map_info_ema = 0., 0., None, 0., 0., None, None

        self.label_buffer = torch.zeros(1000).int().to(self.device)
        self.code_buffer = torch.zeros(1000, self.code_bits).to(self.device)
        self.pred_buffer = torch.zeros(1000, 10).to(self.device)

        self.un_label_buffer = torch.zeros(1000).int().to(self.device)
        self.un_code_buffer = torch.zeros(1000, self.code_bits).to(self.device)
        self.un_pred_buffer = torch.zeros(1000, 10).to(self.device)

        for ep in range(epochs):
            self.epoch = ep
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=ep)
            print("------ Training epochs: {} ------".format(ep))
            self.train(label_data, unlab_data, self.print_freq)
            if scheduler is not None: scheduler.step()
            print("------ Testing epochs: {} ------".format(ep))
            info, n = self.test(query_data, self.print_freq)
            acc = sum(info['lacc'])/n
            if acc > best_acc: best_acc, best_info = acc, info
            if (ep+1) % 2 == 0:
                print("------ Evalution epochs: {} ------".format(ep))
                Map = self.eval(query_data, database, self.code_bits)
                meanAP = Map['meanAP'][0]
                meanAP_ema = Map['meanAP_ema'][0]
                self.writer.add_scalar('meanAP', meanAP, global_step=self.epoch)
                self.writer.add_scalar('meanAP_ema', meanAP_ema, global_step=self.epoch)
                if meanAP > best_map: map_info, best_map = Map, meanAP
                if meanAP_ema > best_map_ema: map_info_ema, best_map_ema = Map, meanAP_ema
            ## save model
            if self.save_freq != 0 and (ep + 1) % self.save_freq == 0:
                self.save(ep)
        print(f">>>[best]", self.gen_info(best_info, n, n, False))
        print(f">>>[best_Map]", self.gen_info(map_info, 1, 1), f"[best_Map_ema]", self.gen_info(map_info_ema, 1, 1))

        self.writer.close()

    def update_ema(self, model, ema_model, alpha, global_step):
        
        alpha = min(1 - 1 / (global_step + 1), alpha)
        
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)


    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u': ubs, 'a': lbs + ubs, 'm': lbs}
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v/n:.3%}' if k[-1]=='c' else f'{k}: {v:.5f}'
            ret.append(s)
        return '\t'.join(ret)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_target = model_out_path / "model_epoch_{}.pth".format(epoch)
            torch.save(state, save_target)
            print('==> save model to {}'.format(save_target))
