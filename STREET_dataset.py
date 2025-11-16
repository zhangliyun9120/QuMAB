import logging
import math
import torch
import shutil
import random
import torch.nn as nn
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import filters
from skimage import transform as skimage_transform
from sklearn.metrics import cohen_kappa_score
from scipy.ndimage import gaussian_filter

from timechat.models.blip2 import Blip2Base, disabled_train
from timechat.models.Qformer import BertConfig, BertLMHeadModel

from torch.cuda.amp import autocast
import einops

import datetime
from sklearn.metrics import accuracy_score


from sklearn.metrics import cohen_kappa_score

import wandb

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision.transforms.functional as TF

from torch.optim import AdamW
from torch.optim import lr_scheduler
import torch.nn.init as init

import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


MAIN_CATEGORIES = ['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly']

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ADDMetric:
    def __init__(self, num_annotators):
        self.num_annotators = num_annotators

    def calculate_agreement_matrix(self, annotations):
        device = annotations[0].device
        agreement_matrix = torch.zeros((self.num_annotators, self.num_annotators), device=device)

        for i in range(self.num_annotators):
            for j in range(i + 1, self.num_annotators):
                labels_i = annotations[i].cpu().numpy()
                labels_j = annotations[j].cpu().numpy()

                kappa = cohen_kappa_score(labels_i, labels_j)

                agreement_matrix[i, j] = agreement_matrix[j, i] = kappa

            agreement_matrix[i, i] = 1.0

        return agreement_matrix

    def calculate_add(self, pred_annotations, true_annotations):
        pred_acm = self.calculate_agreement_matrix(pred_annotations)
        true_acm = self.calculate_agreement_matrix(true_annotations)

        matrix_diff = torch.norm(pred_acm - true_acm, p='fro')

        mask = ~torch.eye(self.num_annotators, dtype=torch.bool, device=pred_acm.device)
        pred_consistency = pred_acm[mask].mean()
        true_consistency = true_acm[mask].mean()

        abs_diff = torch.abs(pred_consistency - true_consistency)
        rel_diff = abs_diff / true_consistency

        return {
            'add_frobenius': matrix_diff.item(),
            'add_absolute': abs_diff.item(),
            'add_relative': rel_diff.item(),
            'pred_consistency': pred_consistency.item(),
            'true_consistency': true_consistency.item(),
            'pred_acm': pred_acm.cpu().numpy(),
            'true_acm': true_acm.cpu().numpy()
        }

    def visualize_agreement_matrices(self, pred_acm, true_acm, save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        sns.heatmap(true_acm, ax=ax1, vmin=0, vmax=1, cmap='YlOrRd')
        ax1.set_title('True Agreement Matrix')

        sns.heatmap(pred_acm, ax=ax2, vmin=0, vmax=1, cmap='YlOrRd')
        ax2.set_title('Predicted Agreement Matrix')

        diff_matrix = np.abs(pred_acm - true_acm)
        sns.heatmap(diff_matrix, ax=ax3, cmap='YlOrRd')
        ax3.set_title('Difference Matrix')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()


class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, category, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.category = category

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = []
        for i in range(1, 11):
            label = self.annotations[f"{self.category}_{i}"].iloc[idx]
            labels.append(label)

        labels = torch.tensor(labels)
        return image, labels


class Backbone(Blip2Base):

    def __init__(
            self,
            vit_model="ckpt/eva-vit-g/eva_vit_g.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            multi_annotation=True,
            q_former_model="ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth",
            freeze_qformer=False,
            num_query_token=10,
            attention_dim=256,
            num_levels=6,
            num_annotators=10
    ):
        super().__init__()

        self.num_annotators = num_annotators

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        self.load_from_pretrained(multi_annotation=multi_annotation, url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, attention_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attention_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_levels)
            ) for _ in range(num_annotators)
        ])

    def forward(self, images):
        device = images.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True,
            )
            frame_cross_attentions = query_output.cross_attentions[-2]
            frame_cross_attentions = frame_cross_attentions[:, :, :, 1:]
            B, H, Q, L = frame_cross_attentions.shape
            frame_cross_attentions = frame_cross_attentions.reshape(B, H, Q, int(np.sqrt(L)), int(np.sqrt(L)))

            image_hidden = query_output.last_hidden_state

            image_features = F.normalize(self.vision_proj(image_hidden), dim=-1)

        features = self.dropout(image_features)

        outputs = []
        for annotator_idx in range(self.num_annotators):
            annotator_feature = features[:, annotator_idx, :]
            classifier = self.output_heads[annotator_idx]
            outputs.append(classifier(annotator_feature))

        outputs = torch.stack(outputs, dim=1)

        return outputs, frame_cross_attentions


def calculate_accuracy(predictions, labels):
    accuracy = (predictions == labels).float().mean()
    return accuracy


def calculate_waf(predictions, labels):
    device = predictions.device
    unique_labels = torch.unique(labels).to(device)
    weights = torch.tensor([(labels == label).float().mean().item() for label in unique_labels], device=device)
    f1_scores = torch.stack([f1_score_torch((labels == label).float(), (predictions == label).float()) for label in unique_labels])
    waf = torch.sum(weights * f1_scores)
    return waf

def f1_score_torch(y_true, y_pred):
    tp = torch.sum(y_true * y_pred).float()
    fp = torch.sum((1 - y_true) * y_pred).float()
    fn = torch.sum(y_true * (1 - y_pred)).float()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1


def calculate_mae(predictions, labels):
    device = predictions.device
    mae = torch.abs(predictions.float() - labels.float()).mean()
    return mae


def calculate_annotator_kappa(all_preds, all_labels, num_annotators, num_classes=6):
    device = all_preds[0].device
    kappas = []

    for i in range(num_annotators):
        for j in range(i + 1, num_annotators):
            labels_i = all_labels[i]
            labels_j = all_labels[j]

            confusion_matrix = torch.zeros((num_classes, num_classes), device=device)
            for k in range(len(labels_i)):
                confusion_matrix[labels_i[k].long(), labels_j[k].long()] += 1

            n_samples = len(labels_i)
            observed_agreement = torch.sum(torch.diag(confusion_matrix)) / n_samples

            row_sum = torch.sum(confusion_matrix, dim=1)
            col_sum = torch.sum(confusion_matrix, dim=0)
            expected_agreement = torch.sum(row_sum * col_sum) / (n_samples * n_samples)

            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement + 1e-8)
            kappas.append(kappa.item())

    return sum(kappas) / len(kappas)


def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train_model(backbone, dataloader_train, optimizer, scheduler, criterion, num_annotators, args, epoch, total_epochs, avg_train_losses):

    backbone.train()

    add_metric = ADDMetric(num_annotators)

    device = next(backbone.parameters()).device
    all_preds = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_annotators)]
    all_labels = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_annotators)]

    total_batches = len(dataloader_train)

    record_losses = [[0] * num_annotators for _ in range(total_batches)]

    for batch_id, batch in enumerate(dataloader_train):
        optimizer.zero_grad()

        images, batch_labels  = batch

        images  = images.cuda(args.local_rank, non_blocking=True)
        batch_labels = batch_labels.cuda(args.local_rank, non_blocking=True)

        all_outputs, _ = backbone(images)

        update_loss = 0

        for annotator_id in range(num_annotators):
            annotator_loss = 0

            label = batch_labels[:, annotator_id]
            output = all_outputs[:, annotator_id, :]
            loss = criterion(output, label)
            annotator_loss += loss

            with torch.no_grad():
                _, preds = torch.max(output, 1)
                all_preds[annotator_id] = torch.cat([all_preds[annotator_id], preds])
                all_labels[annotator_id] = torch.cat([all_labels[annotator_id], label])

            record_losses[batch_id][annotator_id] += annotator_loss.item()
            update_loss += annotator_loss

        if torch.isnan(update_loss):
            print("Loss is NaN!")

        update_loss.backward()

        apply_gradient_clipping(backbone, max_norm=args.max_grad_norm)

        optimizer.step()

        scheduler.step(epoch, batch_id)

    with torch.no_grad():
        avg_losses = [sum(losses) / len(dataloader_train) for losses in zip(*record_losses)]
        total_loss = sum(avg_losses)

        wafs = [calculate_waf(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        total_waf = sum(wafs) / len(wafs)

        maes = [calculate_mae(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        avg_mae = sum(maes) / len(maes)

        accuracys = [calculate_accuracy(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        avg_accuracy = sum(accuracys) / len(accuracys)

        add_results = add_metric.calculate_add(all_preds, all_labels)

        if args.local_rank == 0:
            whole_mean_kappa, whole_std_kappa, whole_kappa_matrix = calculate_and_plot_kappa(
                all_preds, all_labels, num_annotators, args, "train"
            )

            avg_train_losses.append(total_loss)

            log_dict = {
                f"val_{args.category}_loss": total_loss,
                f"val_{args.category}_waf": total_waf,
                f"val_{args.category}_mae": avg_mae,
                f"val_{args.category}_accuracy": avg_accuracy
            }
            log_dict.update({f"train__{args.category}_annotator_{i + 1}_waf": waf for i, waf in enumerate(wafs)})
            log_dict.update({f"train__{args.category}_annotator_{i + 1}_mae": mae for i, mae in enumerate(maes)})

            log_dict.update({
                f"train_{args.category}_add_frobenius": add_results['add_frobenius'],
                f"train_{args.category}_add_absolute": add_results['add_absolute'],
                f"train_{args.category}_add_relative": add_results['add_relative'],
                f"train_{args.category}_pred_consistency": add_results['pred_consistency'],
                f"train_{args.category}_true_consistency": add_results['true_consistency']
            })
            wandb.log(log_dict)

            if (epoch + 1) % args.save_epoch == 0:
                add_metric.visualize_agreement_matrices(
                    add_results['pred_acm'],
                    add_results['true_acm'],
                    f'{args.save_path}/train_agreement_matrices_epoch_{epoch}.png'
                )

            print(f"Epoch {epoch + 1}/{total_epochs}, train_total_loss: {total_loss:.4f}")
            for i, loss in enumerate(avg_losses):
                print(f"Annotator {i + 1}, Train Avg_loss: {loss:.4f}")
            for i, waf in enumerate(wafs):
                print(f"Annotator {i + 1}, Train Avg_waf: {waf:.4f}")
            for i, mae in enumerate(maes):
                print(f"Annotator {i + 1}, Train Avg_mae: {mae:.4f}")
            for i, accuracy in enumerate(accuracys):
                print(f"Annotator {i + 1}, Train Avg_accuracy: {accuracy:.4f}")

            print(f"train_{args.category}_add_frobenius: {add_results['add_frobenius']}")
            print(f"train_{args.category}_add_absolute: {add_results['add_absolute']}")
            print(f"train_{args.category}_add_relative: {add_results['add_relative']}")
            print(f"train_{args.category}_pred_consistency: {add_results['pred_consistency']}")
            print(f"train_{args.category}_true_consistency: {add_results['true_consistency']}")

        return avg_train_losses


def evaluate_model(backbone, dataloader_test, criterion, num_annotators, args, epoch, total_epochs, avg_valid_losses):
    backbone.eval()

    add_metric = ADDMetric(num_annotators)

    with torch.no_grad():
        device = next(backbone.parameters()).device
        all_preds = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_annotators)]
        all_labels = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_annotators)]

        total_batches = len(dataloader_test)

        record_losses = [[0] * num_annotators for _ in range(total_batches)]

        for batch_id, batch in enumerate(dataloader_test):

            images, batch_labels = batch

            images = images.cuda(args.local_rank, non_blocking=True)
            batch_labels = batch_labels.cuda(args.local_rank, non_blocking=True)

            all_outputs, _ = backbone(images)

            for annotator_id in range(num_annotators):
                annotator_loss = 0

                label = batch_labels[:, annotator_id]
                output = all_outputs[:, annotator_id, :]
                loss = criterion(output, label)
                annotator_loss += loss

                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    all_preds[annotator_id] = torch.cat([all_preds[annotator_id], preds])
                    all_labels[annotator_id] = torch.cat([all_labels[annotator_id], label])

                record_losses[batch_id][annotator_id] += annotator_loss.item()

        avg_losses = [sum(losses) / len(dataloader_test) for losses in zip(*record_losses)]
        total_loss = sum(avg_losses)

        wafs = [calculate_waf(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        total_waf = sum(wafs) / len(wafs)

        maes = [calculate_mae(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        avg_mae = sum(maes) / len(maes)

        accuracys = [calculate_accuracy(all_preds[a], all_labels[a]) for a in range(num_annotators)]
        avg_accuracy = sum(accuracys) / len(accuracys)

        add_results = add_metric.calculate_add(all_preds, all_labels)

        if args.local_rank == 0:
            whole_mean_kappa, whole_std_kappa, whole_kappa_matrix = calculate_and_plot_kappa(
                all_preds, all_labels, num_annotators, args, "val"
            )

            avg_valid_losses.append(total_loss)

            if not args.evaluate:
                # wandb记录
                log_dict = {
                    f"val_{args.category}_loss": total_loss,
                    f"val_{args.category}_waf": total_waf,
                    f"val_{args.category}_mae": avg_mae,
                    f"val_{args.category}_accuracy": avg_accuracy
                }
                # 记录每个标注者的详细指标
                log_dict.update({f"val_{args.category}_annotator_{i + 1}_waf": waf for i, waf in enumerate(wafs)})
                log_dict.update({f"val_{args.category}_annotator_{i + 1}_mae": mae for i, mae in enumerate(maes)})

                log_dict.update({
                    f"val_{args.category}_add_frobenius": add_results['add_frobenius'],
                    f"val_{args.category}_add_absolute": add_results['add_absolute'],
                    f"val_{args.category}_add_relative": add_results['add_relative'],
                    f"val_{args.category}_pred_consistency": add_results['pred_consistency'],
                    f"val_{args.category}_true_consistency": add_results['true_consistency']
                })
                wandb.log(log_dict)

            if (epoch + 1) % args.save_epoch == 0:
                add_metric.visualize_agreement_matrices(
                    add_results['pred_acm'],
                    add_results['true_acm'],
                    f'{args.save_path}/avl_agreement_matrices_epoch_{epoch}.png'
                )

            print(f"Epoch {epoch + 1}/{total_epochs}, test_total_loss: {total_loss:.4f}")
            for i, loss in enumerate(avg_losses):
                print(f"Annotator {i + 1}, Val Avg_loss: {loss:.4f}")
            for i, waf in enumerate(wafs):
                print(f"Annotator {i + 1}, Val Avg_waf: {waf:.4f}")
            for i, mae in enumerate(maes):
                print(f"Annotator {i + 1}, Val Avg_mae: {mae:.4f}")
            for i, accuracy in enumerate(accuracys):
                print(f"Annotator {i + 1}, Val Avg_accuracy: {accuracy:.4f}")

            print(f"avl_{args.category}_add_frobenius: {add_results['add_frobenius']}")
            print(f"avl_{args.category}_add_absolute: {add_results['add_absolute']}")
            print(f"avl_{args.category}_add_relative: {add_results['add_relative']}")
            print(f"avl_{args.category}_pred_consistency: {add_results['pred_consistency']}")
            print(f"avl_{args.category}_true_consistency: {add_results['true_consistency']}")

        return avg_valid_losses, total_loss


def calculate_kappa_matrix(all_preds, all_labels, num_annotators):
    device = all_preds[0].device
    kappa_matrix = torch.zeros((num_annotators, num_annotators), device=device)

    for i in range(num_annotators):
        for j in range(num_annotators):
            if i == j:
                kappa_matrix[i, j] = 1.0
            else:
                preds = all_preds[i].cpu().numpy()
                labels = all_labels[j].cpu().numpy()

                if len(preds) == len(labels):
                    kappa = cohen_kappa_score(preds, labels)
                    kappa_matrix[i, j] = kappa
                else:
                    print(f"Warning: Length mismatch for annotators {i} and {j}. Skipping Kappa calculation.")

    mask = ~torch.eye(num_annotators, dtype=torch.bool, device=device)
    mean_kappa = kappa_matrix[mask].mean().item()
    std_kappa = kappa_matrix[mask].std().item()

    return mean_kappa, std_kappa, kappa_matrix.cpu().numpy()


def plot_kappa_heatmap(kappa_matrix, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(kappa_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title("Inter-Annotator Kappa Agreement with Model Predictions")
    plt.xlabel("Annotator ID")
    plt.ylabel("Annotator ID")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def calculate_and_plot_kappa(all_preds, all_labels, num_annotators, args, phase):
    device = all_preds[0].device
    kappa_matrix = torch.zeros((num_annotators, num_annotators), device=device)

    for i in range(num_annotators):
        for j in range(num_annotators):
            if i == j:
                kappa_matrix[i, j] = 1.0
                continue

            # 获取两个标注者的标签
            labels_i = all_labels[i]
            labels_j = all_labels[j]

            confusion_matrix = torch.zeros((6, 6), device=device)  # 假设6个等级
            for k in range(len(labels_i)):
                confusion_matrix[labels_i[k].long(), labels_j[k].long()] += 1

            n_samples = len(labels_i)
            observed_agreement = torch.sum(torch.diag(confusion_matrix)) / n_samples

            row_sum = torch.sum(confusion_matrix, dim=1)
            col_sum = torch.sum(confusion_matrix, dim=0)
            expected_agreement = torch.sum(row_sum * col_sum) / (n_samples * n_samples)

            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement + 1e-8)
            kappa_matrix[i, j] = kappa.item()

    if args.local_rank == 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(kappa_matrix.cpu().numpy(),
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    vmin=0,
                    vmax=1.0)
        plt.title(f'Inter-annotator Agreement - {args.category}')
        plt.xlabel('Annotator ID')
        plt.ylabel('Annotator ID')

        plt.savefig(os.path.join(args.save_path, f'{phase}_kappa_heatmap_{args.category.lower()}.png'))
        plt.show()
        plt.close()

    mask = ~torch.eye(num_annotators, dtype=torch.bool, device=device)
    mean_kappa = kappa_matrix[mask].mean()
    std_kappa = kappa_matrix[mask].std()

    return mean_kappa.item(), std_kappa.item(), kappa_matrix.cpu().numpy()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_weights(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))


def get_scheduler(optimizer, args, iterations=-1):
    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=iterations)
    else:
        scheduler = None  # constant scheduler
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def apply_gradient_clipping(backbone, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=max_norm)  # 梯度裁剪


def print_total_params(model):
    if isinstance(model, DDP):
        model = model.module
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_mb = total_params * 4 / (1024 ** 2)
    total_trainable_params_mb = total_trainable_params * 4 / (1024 ** 2)
    return {
        'total': round(total_params_mb, 2),
        'trainable': round(total_trainable_params_mb, 2)
    }


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.total_steps = max_epoch * iters_per_epoch

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            self.warmup_lr_schedule(total_cur_step)
        else:
            self.cosine_lr_schedule(total_cur_step)

    def cosine_lr_schedule(self, step):
        """Decay the learning rate"""
        progress = (step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        lr = (self.init_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + self.min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def warmup_lr_schedule(self, step):
        """Warmup the learning rate"""
        lr = min(self.init_lr, self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * step / max(self.warmup_steps, 1))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def main(args):
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        wandb.init(project='Street_Classifier',  name='QFormer')

    train_dataset = ImageDataset(args.img_dir_train, args.csv_file_train, args.category, transform=data_transform)
    train_sampler = DistributedSampler(dataset=train_dataset)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True, sampler=train_sampler, num_workers=8)
    val_dataset = ImageDataset(args.img_dir_val, args.csv_file_val, args.category, transform=data_transform)
    val_sampler = DistributedSampler(dataset=val_dataset)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=args.batch_size, pin_memory=True, sampler=val_sampler, num_workers=8)
    if args.evaluate:
        test_dataset = ImageDataset(args.img_dir_test, args.csv_file_test, args.category, transform=data_transform)
        test_sampler = DistributedSampler(dataset=test_dataset)
        dataloader_test = DataLoader(dataset=test_dataset, batch_size=args.batch_size, pin_memory=True, sampler=test_sampler, num_workers=8)

    attention_dim = 256
    feature_dim = 768
    NUM_LEVELS = 6
    NUM_ANNOTATORS = 10

    backbone = Backbone(vit_model=args.vit_model, q_former_model=args.q_former_model, attention_dim=attention_dim,
                        num_levels=NUM_LEVELS, num_annotators=NUM_ANNOTATORS).cuda(local_rank)
    backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(dataloader_train) * args.epochs
    warmup_steps = int(0.2 * total_steps)
    min_lr = 1e-7
    max_lr = 1e-5
    warmup_start_lr = max_lr / 10

    optimizer = AdamW(
        list(backbone.parameters()),
        lr=max_lr,
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=args.epochs,
        iters_per_epoch=len(dataloader_train),
        min_lr=min_lr,
        init_lr=max_lr,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
    )

    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    avg_train_losses, avg_valid_losses = [], []

    if args.evaluate:
        if args.early_stopped:
            checkpoint_path = f'{args.save_path}/checkpoint_best_earlystopping.pth'
            checkpoint = load_checkpoint(checkpoint_path, args)
            backbone.module.load_state_dict(checkpoint)
        else:
            checkpoint_path = f'{args.save_path}/{args.save_model}'

            checkpoint = load_checkpoint(checkpoint_path, args)

            backbone.module.load_state_dict(checkpoint['state_dict_backbone'])

        if local_rank == 0 and args.process_directory:
            image_paths = get_image_paths(args.visualization_save_dir)
            if not image_paths:
                print(f"No images found in {args.visualization_save_dir}")
                return

            ground_truth_dict = load_ground_truth(args.csv_file_test)

            results_dir = os.path.join(os.path.dirname(args.visualization_save_dir), 'visualization_results_normal_gradcam')

            visualize_specific_images(
                backbone.module,
                image_paths,
                results_dir,
                ground_truth_dict
            )

        _, _ = evaluate_model(backbone, dataloader_test, criterion, NUM_ANNOTATORS, args, -1, -1, avg_valid_losses)

        wandb.finish()
        return

    initial_epoch = 0
    total_epochs = args.epochs
    if local_rank == 0:
        patience = 25
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001)

    try:
        for epoch in range(initial_epoch, total_epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            avg_train_losses = train_model(backbone, dataloader_train, optimizer, scheduler, criterion, NUM_ANNOTATORS, args, epoch, total_epochs, avg_train_losses)
            avg_valid_losses, valid_loss = evaluate_model(backbone, dataloader_val, criterion, NUM_ANNOTATORS, args, epoch, total_epochs, avg_valid_losses)

            if local_rank == 0:
                early_stopping(valid_loss, backbone.module)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            torch.cuda.empty_cache()

            if (epoch + 1) % args.save_epoch == 0 and local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict_backbone': backbone.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, f'{args.save_path}/checkpoint_epoch_{epoch}.pth')
                print(f"Model saved to {args.save_path}/checkpoint_epoch_{epoch}.pth !!!")

        if local_rank == 0:
            backbone.module.load_state_dict(torch.load('checkpoint.pt'))
            print(f"Load and Save Best performance model parameters of EarlyStopping！！！")
            torch.save(backbone.module.state_dict(), f'{args.save_path}/checkpoint_best_earlystopping.pth')
            print(f"Saved final model to {args.save_path}/checkpoint_best_earlystopping.pth")

            save_loss_to_file(avg_train_losses, avg_valid_losses, f'{args.save_path}/losses.csv')

            wandb.finish()
    except KeyboardInterrupt:
        print("Training interrupted by user keyboardInterrupt!!!")
    finally:
        if local_rank == 0:
            wandb.finish()


def visualize_attention(image, attention_map, title):
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap * 0.4 + image

    superimposed_img = superimposed_img / superimposed_img.max()

    return superimposed_img


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def process_directory_images(backbone, image_dir, results_dir=None):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(supported_formats)]

    if not image_paths:
        print(f"No supported images found in {image_dir}")
        return

    if results_dir is None:
        parent_dir = os.path.dirname(image_dir.rstrip('/'))
        results_dir = os.path.join(parent_dir, 'visualization_results')

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    visualize_specific_images(backbone, image_paths, results_dir, transform)
    print(f"\nVisualization completed! Results saved to {results_dir}")


def get_image_paths(directory):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []

    for file in os.listdir(directory):
        if file.lower().endswith(supported_formats):
            image_paths.append(os.path.join(directory, file))

    return sorted(image_paths)


def load_ground_truth(csv_file):
    df = pd.read_csv(csv_file)
    label_dict = {}

    for _, row in df.iterrows():
        img_name = row['name']
        label_dict[img_name] = {}
        for cat in ['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly']:
            label_dict[img_name][cat] = {}
            for i in range(1, 11):
                label_dict[img_name][cat][i - 1] = row[f"{cat}_{i}"]

    return label_dict


def visualize_specific_images(backbone, image_paths, save_dir, ground_truth_dict, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    backbone.eval()
    device = next(backbone.parameters()).device
    categories = ['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly']

    os.makedirs(save_dir, exist_ok=True)
    formatted_results = []

    with torch.no_grad():
        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            original_image = Image.open(img_path).convert('RGB')
            image_tensor = transform(original_image).unsqueeze(0).to(device)
            all_outputs, frame_cross_attentions = backbone(image_tensor)

            plt.close('all')
            fig, axs = plt.subplots(2, 6, figsize=(30, 12))
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

            axs[0, 0].imshow(original_image.resize((224, 224)))
            axs[0, 0].set_title("Original Image", fontsize=12, pad=10)
            axs[0, 0].axis('off')

            prediction_differences = []

            norm_img = np.float32(original_image.resize((224, 224))) / 255

            for ann_idx in range(10):
                if ann_idx < 5:
                    row, col = 0, ann_idx + 1
                else:
                    row, col = 1, ann_idx - 5 + 1

                curr_attention = frame_cross_attentions[0][:, ann_idx, :, :]
                head_attentions = []
                for head_idx in range(12):
                    head_att = curr_attention[head_idx]
                    head_att = head_att.reshape(16, 16)
                    head_att = (head_att - head_att.min()) / (head_att.max() - head_att.min() + 1e-8)
                    head_attentions.append(head_att)
                annotator_attention = torch.stack(head_attentions, dim=0)
                attention = torch.mean(annotator_attention, dim=0).cpu()

                gradcam_image = getAttMap(norm_img, attention.numpy().astype(np.float32), blur=True)

                axs[row, col].imshow(gradcam_image)
                axs[row, col].set_title(f'Annotator {ann_idx + 1}', fontsize=12, pad=10)
                axs[row, col].axis('off')

                for cat_idx, category in enumerate(categories):
                    output = all_outputs[cat_idx][0, ann_idx, :]
                    prob = F.softmax(output, dim=0)[1].item()
                    pred = 1 if prob > 0.5 else 0
                    gt = ground_truth_dict[img_name][category][ann_idx]

                    if abs(pred - gt) > 0:
                        prediction_differences.append({
                            'annotator': ann_idx + 1,
                            'category': category,
                            'pred': pred,
                            'prob': prob,
                            'gt': gt,
                            'diff': abs(prob - (1 if gt == 1 else 0))
                        })

            prediction_differences.sort(key=lambda x: x['diff'], reverse=True)

            axs[1, 0].axis('off')
            summary_text = []
            for diff in prediction_differences[:5]:
                summary_text.append(
                    f"Ann.{diff['annotator']} ({diff['category']}): "
                    f"Pred {diff['pred']}({diff['prob']:.3f})/GT {diff['gt']}"
                )

            axs[1, 0].text(0.5, 0.5, '\n'.join(summary_text),
                           transform=axs[1, 0].transAxes,
                           fontsize=12,
                           horizontalalignment='center',
                           verticalalignment='center')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"analysis_{img_name}"), dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()
            plt.close()

            for category in categories:
                result_row = {
                    'Image': img_name,
                    'Category': category,
                    'Measure': 'Predictions/Ground Truth'
                }

                for ann_idx in range(10):
                    output = all_outputs[categories.index(category)][0, ann_idx, :]
                    prob = F.softmax(output, dim=0)[1].item()
                    pred = 1 if prob > 0.5 else 0
                    gt = ground_truth_dict[img_name][category][ann_idx]
                    result_row[f'Annotator_{ann_idx + 1}'] = f"{pred} ({prob:.3f})/{gt}"

                formatted_results.append(result_row)

            print(f"Processed {img_name}")

    results_df = pd.DataFrame(formatted_results)
    results_df.to_csv(os.path.join(save_dir, "prediction_results.csv"), index=False)

    return results_df


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap

def save_loss_to_file(train_losses, valid_losses, filename='Try_street/checkpoints/losses.csv'):
    loss_df = pd.DataFrame({
        'Train_Loss': train_losses,
        'Validation_Loss': valid_losses
    })

    loss_df.to_csv(filename, index=False)
    print(f"Losses saved to {filename}")


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.local_rank}')
    return checkpoint

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def print_distributions(df, num_annotators, prefix='discrete'):
    for i in range(1, num_annotators + 1):
        column_name = f'{prefix}_{i}'
        if column_name in df.columns:
            print(f"Distribution for {column_name}:")
            print(df[column_name].value_counts(normalize=True))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Process for Multi-annotation Classification")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument('--img_dir_train', type=str, default="datasets/annotated/normal/images_train", help='image')
    parser.add_argument('--csv_file_train', type=str, default="datasets/annotated/normal/train_data.csv", help='label path')
    parser.add_argument('--img_dir_val', type=str, default="datasets/annotated/normal/images_val", help='image')
    parser.add_argument('--csv_file_val', type=str, default="datasets/annotated/normal/val_data.csv", help='label path')
    parser.add_argument('--img_dir_test', type=str, default="datasets/annotated/normal/images_test", help='image')
    parser.add_argument('--csv_file_test', type=str, default="datasets/annotated/normal/test_data.csv", help='label path')

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)

    parser.add_argument('--vit_model', type=str, default="ckpt/eva-vit-g/eva_vit_g.pth")
    parser.add_argument('--q_former_model', type=str, default="ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth")

    parser.add_argument('--batch_size', type=int, default=64, help="batch_size.")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_policy', type=str, default='custom', help="warmup, plateau, step, custom")
    parser.add_argument('--step_size', type=int, default=5000, help="how often to decay learning rate")
    parser.add_argument('--beta1', type=int, default=0.9, help='Adam parameter')
    parser.add_argument('--beta2', type=int, default=0.999, help='Adam parameter')
    parser.add_argument('--init', type=str, default='kaiming', help="initialization [gaussian/kaiming/xavier/orthogonal")
    parser.add_argument('--gamma', type=float, default=0.5, help="how much to decay learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum norm for gradient clipping')

    parser.add_argument('--epochs', type=int, default=200, help="specify the gpu to load the model.")
    parser.add_argument('--save_epoch', type=int, default=10, help="specify the gpu to load the model.")
    parser.add_argument('--save_path', type=str, default="Try_street/checkpoints/normal_qformer_class/Safe_200", help='')
    parser.add_argument('--save_model', type=str, default="final.pth", help='')

    parser.add_argument('--early_stopped', type=bool, default=False)

    parser.add_argument('--caculate_cross_annotators', type=bool, default=False)

    parser.add_argument('--visualize_annotator_attention', type=bool, default=False)

    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")

    parser.add_argument('--category', type=str, default='Safe',
                            choices=['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly'],
                            help='Specify which category to train')

    parser.add_argument('--process_directory', type=bool, default=False,
                        help='Whether to process all images in the directory')
    parser.add_argument('--run_evaluation', type=bool, default=False,
                        help='Whether to run evaluation on test dataset')
    parser.add_argument('--visualization_save_dir', type=str,
                        default='Try_street/analysis_qformer/specific_images_normal',
                        help='Directory containing images to process')

    args = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    set_seed(42)

    main(args)