import pickle
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from .metrics import eval_result
import math
from .utils import to_device


class Trainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None, model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        if self.args.do_train:
            self.before_multimodal_train()
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = to_device(batch, self._device)

                    (mu, std), logits1, logits2, labels, topic_loss = self._step(batch, mode="train", step=self.step)
                    GIB_loss = self.loss_func(logits1, labels.view(-1))
                    task_loss = self.loss_func(logits2, labels.view(-1))
                    if self.args.is_IB:
                        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                        GIB_loss += self.args.beta * KL_loss
                    self.writer.add_scalar(tag='task_loss', scalar_value=task_loss.detach().cpu().item(),
                                           global_step=self.step)
                    self.writer.add_scalar(tag='GIB_loss', scalar_value=GIB_loss.detach().cpu().item(),
                                           global_step=self.step)
                    self.writer.add_scalar(tag='topic_loss', scalar_value=topic_loss.detach().cpu().item(),
                                           global_step=self.step)
                    loss = task_loss + self.args.eta1 * GIB_loss + self.args.eta2 * topic_loss
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer is not None:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)  # tensorbordx
                        avg_loss = 0
                    if self.step % 50 == 0:
                        self.evaluate(self.step)

                if epoch >= self.args.eval_begin_epoch:
                    self.test(epoch)
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = to_device(batch, self._device)  # to cpu/cuda device
                    (mu, std), logits1, logits2, labels, topic_loss = self._step(batch, mode="dev")
                    GIB_loss = self.loss_func(logits1, labels.view(-1))
                    task_loss = self.loss_func(logits2, labels.view(-1))
                    if self.args.is_IB:
                        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                        GIB_loss += self.args.beta * KL_loss
                    loss = task_loss + self.args.eta1 * GIB_loss + self.args.eta2 * topic_loss
                    total_loss += loss.detach().cpu().item()

                    preds = logits2.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / len(self.test_data),
                                           global_step=epoch)  # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path + "/best_model.pth"))
            self.model.load_state_dict(torch.load(self.args.load_path + "/best_model.pth"))
            self.logger.info("Load model successful!")
            self.model.to(self.args.device)
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = to_device(batch, self._device)  # to cpu/cuda device
                    (mu, std), logits1, logits2, labels, topic_loss = self._step(batch, mode="dev")
                    GIB_loss = self.loss_func(logits1, labels.view(-1))
                    task_loss = self.loss_func(logits2, labels.view(-1))
                    if self.args.is_IB:
                        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
                        GIB_loss += self.args.beta * KL_loss
                    loss = task_loss + self.args.eta1 * GIB_loss + self.args.eta2 * topic_loss
                    total_loss += loss.detach().cpu().item()

                    preds = logits2.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data),
                                           global_step=epoch)  # tensorbordx
                total_loss = 0
                ############
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}" \
                                 .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.best_test_metric = micro_f1
                    self.best_test_epoch = epoch

        self.model.train()

    def _step(self, batch, mode="train", step=0):
        (mu, std), logits1, logits2, topic_loss = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], piece2word=batch["piece2word"],
                head_tail_pos=batch["head_tail_pos"],  head_object_tokens=batch["head_object_tokens"], tail_object_tokens=batch["tail_object_tokens"],
                t_objects_tokens=batch["t_objects_tokens"], t_attributes_tokens=batch["t_attributes_tokens"], t_relations_tokens=batch["t_relations_tokens"], 
                v_objects_tokens=batch["v_objects_tokens"],  v_attributes_tokens=batch["v_attributes_tokens"], v_relations_tokens=batch["v_relations_tokens"],
                TSG_adj_matrix=batch["TSG_adj_matrix"], TSG_edge_mask=batch["TSG_edge_mask"], VSG_adj_matrix=batch["VSG_adj_matrix"], VSG_edge_mask=batch["VSG_edge_mask"],
                labels=batch["re_label"], X_T_bow=batch["X_T_bow"], X_V_bow=batch["X_V_bow"], writer=self.writer, step=step)

        return (mu, std), logits1, logits2, batch["re_label"], topic_loss

    def before_multimodal_train(self):
        pretrained_params = []
        main_params = []
        for name, param in self.model.named_parameters():
            if 'text_model' in name:
                pretrained_params.append(param)
            elif 'vision_model' in name:
                pretrained_params.append(param)
            else:
                main_params.append(param)
        optimizer_grouped_parameters = [
            {'params': pretrained_params, 'lr': self.args.lr_pretrained, 'weight_decay': 1e-2},
            {'params': main_params, 'lr': self.args.lr_main, 'weight_decay': 1e-2},
        ]

        self.optimizer = optim.Adam(optimizer_grouped_parameters)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
