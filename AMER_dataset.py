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

from timechat.processors.video_processor import ToTHWC, ToUint8, load_video_with_resample
from timechat.processors import transforms_video, AlproVideoTrainProcessor
from timechat.models.blip2 import Blip2Base, disabled_train
from timechat.models.Qformer import BertConfig, BertLMHeadModel

from torch.cuda.amp import autocast
import einops

import datetime
from sklearn.metrics import accuracy_score
from skimage import transform as skimage_transform
from sklearn.metrics import cohen_kappa_score
from scipy.ndimage import gaussian_filter

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


EMOTIONS = ['worried', 'happy', 'neutral', 'angry', 'surprise', 'sad', 'other', 'unknown']

transform = AlproVideoTrainProcessor(image_size=224).transform

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

                valid_idx = (labels_i != -1) & (labels_j != -1)
                if np.any(valid_idx):
                    kappa = cohen_kappa_score(labels_i[valid_idx], labels_j[valid_idx])

                    agreement_matrix[i, j] = kappa
                    agreement_matrix[j, i] = kappa

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


class VideoDataset(Dataset):
    def __init__(self, video_dir, csv_file):
        self.video_dir = video_dir
        self.df = pd.read_csv(csv_file)
        self.num_annotators = sum(1 for col in self.df.columns if col.startswith('discrete'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = f"{row['name']}.mp4"
        video_path = os.path.join(self.video_dir, video_name)

        video, _ = load_video_with_resample(
            video_path=video_path,
            n_frms=96,
            height=224,
            width=224,
            return_msg=True
        )
        video = transform(video)

        original_frames = video.size(1)

        emotions = [row[f'discrete{i + 1}'] for i in range(self.num_annotators)]
        emotion_labels = torch.zeros(self.num_annotators, len(EMOTIONS))
        valid_mask = torch.zeros(self.num_annotators, dtype=torch.bool)
        for i, emotion in enumerate(emotions):
            if pd.notna(emotion) and emotion in EMOTIONS:
                emotion_labels[i, EMOTIONS.index(emotion)] = 1
                valid_mask[i] = True

        return video, emotion_labels, valid_mask, original_frames


def dynamic_pad_collate_fn(batch):
    videos, emotion_labels, valid_masks, original_frames = zip(*batch)

    max_frames = max(video.size(1) for video in videos)

    padded_videos = []
    for video in videos:
        pad_size = max_frames - video.size(1)
        if pad_size > 0:
            padding = torch.zeros((video.size(0), pad_size, video.size(2), video.size(3)))
            padded_video = torch.cat([video, padding], dim=1)
        else:
            padded_video = video
        padded_videos.append(padded_video)

    padded_videos = torch.stack(padded_videos)
    emotion_labels = torch.stack(emotion_labels)
    valid_masks = torch.stack(valid_masks)
    original_frames = torch.tensor(original_frames)

    return padded_videos, emotion_labels, valid_masks, original_frames


class Backbone(Blip2Base):

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

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
            num_query_token=32,
            video_q_former_model="ckpt/Video-LLaMA-2-7B-Finetuned/VL_LLaMA_2_7B_Finetuned.pth",
            max_frame_pos=96,
            frozen_video_Qformer=False,
            num_video_query_token=13,
            attention_dim=256,
            feature_dim=768,
            num_classes=8,
            num_annotators=13
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

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None
        self.load_from_pretrained(multi_annotation=multi_annotation,
                                  url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token,
                                                                              vision_width=self.Qformer.config.hidden_size)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if os.path.isfile(video_q_former_model):
            ckpt = torch.load(video_q_former_model, map_location="cpu")
            old_frame_pos_embed_size = ckpt['model']['video_frame_position_embedding.weight'].size()
            new_frame_pos_embed_size = self.video_frame_position_embedding.weight.size()
            if old_frame_pos_embed_size != new_frame_pos_embed_size:
                from timechat.processors.video_processor import interpolate_frame_pos_embed
                print(
                    f'video_frame_position_embedding size is not the same, interpolate from {old_frame_pos_embed_size} to {new_frame_pos_embed_size}')
                ckpt['model']['video_frame_position_embedding.weight'] = interpolate_frame_pos_embed(
                    ckpt['model']['video_frame_position_embedding.weight'], new_n_frm=new_frame_pos_embed_size[0])
            if multi_annotation:
                if 'video_query_tokens' in ckpt['model']:
                    old_video_query_tokens_size = ckpt['model']['video_query_tokens'].size()
                    new_video_query_tokens_size = self.video_query_tokens.size()
                    if old_video_query_tokens_size != new_video_query_tokens_size:
                        print(
                            f'query_tokens size is not the same, interpolate from {old_video_query_tokens_size} to {new_video_query_tokens_size}')
                        old_video_query_tokens_reshaped = ckpt['model']['video_query_tokens'].squeeze(
                            0)
                        new_video_query_tokens_interpolated = interpolate_frame_pos_embed(
                            old_video_query_tokens_reshaped, new_n_frm=new_video_query_tokens_size[1])
                        ckpt['model']['video_query_tokens'] = new_video_query_tokens_interpolated.unsqueeze(
                            0)
            msg = self.load_state_dict(ckpt['model'], strict=False)
            print("Load Checkpoint from: {}".format(video_q_former_model))
        else:
            raise RuntimeError("checkpoint video_q_former_model path is invalid")

        if frozen_video_Qformer:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        self.vision_proj = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.Dropout(0.3)
        )

        self.dropout = nn.Dropout(0.5)

        self.emotion_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attention_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ) for _ in range(num_annotators)
        ])

        self.vision_proj.apply(init_weights(args.init))
        self.emotion_classifier.apply(init_weights(args.init))

    def forward(self, unpadded_videos):
        video_hiddens = []
        video_cross_attentions = []

        for x in unpadded_videos:
            x = x.unsqueeze(0)
            batch_size, channel, time_length, height, width = x.size()
            device = x.device
            x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(x)).to(device)

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    output_attentions=False,
                    return_dict=True,
                )

                position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                q_hidden_state = query_output.last_hidden_state

                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
                frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
                frame_hidden_state = frame_position_embeddings + frame_hidden_state

                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    output_attentions=True,
                    return_dict=True,
                )
                video_cross_attention = video_query_output.cross_attentions[10]
                video_hidden = video_query_output.last_hidden_state
                video_hidden = F.normalize(self.vision_proj(video_hidden), dim=-1)

            video_hiddens.append(video_hidden.squeeze(0))
            video_cross_attentions.append(video_cross_attention.squeeze(0))

        video_hiddens = torch.stack(video_hiddens)
        video_cross_attentions = torch.stack(video_cross_attentions)
        features = self.dropout(video_hiddens)

        outputs = []
        for annotator_idx in range(self.num_annotators):
            annotator_feature = features[:, annotator_idx, :]
            classifier = self.emotion_classifier[annotator_idx]
            outputs.append(classifier(annotator_feature))

        emotion_logits = torch.stack(outputs, dim=1)

        return emotion_logits, video_cross_attentions


def calculate_accuracy(predictions, labels):
    accuracy = (predictions == labels).float().mean()
    return accuracy


def calculate_waf(predictions, labels):
    device = predictions.device
    unique_labels = torch.unique(labels).to(device)

    if len(unique_labels) == 0:
        return 0.0

    weights = torch.tensor([(labels == label).float().mean().item() for label in unique_labels], device=device)
    f1_scores = torch.stack(
        [f1_score_torch((labels == label).float(), (predictions == label).float()) for label in unique_labels])
    waf = torch.sum(weights * f1_scores)

    return waf.item()

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

        padded_videos, emotion_labels, valid_masks, original_frames = batch

        padded_videos = padded_videos.cuda(args.local_rank, non_blocking=True)
        emotion_labels = emotion_labels.cuda(args.local_rank, non_blocking=True)
        valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)
        original_frames = original_frames.cuda(args.local_rank, non_blocking=True)

        unpadded_videos = []
        for i in range(len(padded_videos)):
            video = padded_videos[i]
            original_frame_count = original_frames[i].item()
            if original_frame_count < video.size(1):
                video = video[:, :original_frame_count]
            unpadded_videos.append(video)

        emotion_logits, _ = backbone(unpadded_videos)

        update_loss = 0

        for annotator_id in range(num_annotators):
            valid_samples = valid_masks[:, annotator_id]
            annotator_loss = torch.tensor(0.0, device=valid_samples.device)
            valid_count = 0
            for sample_idx, is_valid in enumerate(valid_samples):
                if is_valid:
                    sample_logit = emotion_logits[sample_idx, annotator_id, :].unsqueeze(0)
                    sample_target = torch.argmax(emotion_labels[sample_idx, annotator_id]).unsqueeze(0)
                    sample_loss = criterion(sample_logit, sample_target)
                    annotator_loss += sample_loss
                    valid_count += 1
                    with torch.no_grad():
                        _, sample_pred = torch.max(sample_logit, 1)
                        all_preds[annotator_id] = torch.cat([all_preds[annotator_id], sample_pred])
                        all_labels[annotator_id] = torch.cat([all_labels[annotator_id], sample_target])
                else:
                    all_preds[annotator_id] = torch.cat(
                        [all_preds[annotator_id], torch.tensor([-1], device=all_preds[annotator_id].device)])
                    all_labels[annotator_id] = torch.cat(
                        [all_labels[annotator_id], torch.tensor([-1], device=all_preds[annotator_id].device)])
            if valid_count:
                annotator_loss /= valid_count
            record_losses[batch_id][annotator_id] += annotator_loss.item()
            update_loss += annotator_loss

        if update_loss > 0:
            if torch.isnan(update_loss):
                print("Loss is NaN!")
            update_loss.backward()

            apply_gradient_clipping(backbone, max_norm=args.max_grad_norm)

            optimizer.step()

            scheduler.step(epoch, batch_id)

    with torch.no_grad():
        avg_losses = []
        for annotator_losses in zip(*record_losses):
            valid_losses = [loss for loss in annotator_losses if loss > 0]
            if valid_losses:
                avg_losses.append(sum(valid_losses) / len(valid_losses))
            else:
                avg_losses.append(0.0)
        total_loss = sum(avg_losses)

        avg_wafs = [calculate_waf(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        total_waf = sum(avg_wafs) / len(avg_wafs)

        maes = [calculate_mae(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        avg_mae = sum(maes) / len(maes)

        accuracys = [calculate_accuracy(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        # avg_accuracy = sum(accuracys) / len(accuracys)

        add_results = add_metric.calculate_add(all_preds, all_labels)

        if args.local_rank == 0:
            whole_mean_kappa, whole_std_kappa, whole_kappa_matrix = calculate_and_plot_kappa(
                all_preds, all_labels, num_annotators, args, "train"
            )

            avg_train_losses.append(total_loss)

            log_dict = {
                "train_total_loss": total_loss,
                "train_total_waf": total_waf
            }
            log_dict.update({f"train_annotator_{i + 1}_avg_loss": loss for i, loss in enumerate(avg_losses)})
            log_dict.update({f"train_annotator_{i + 1}_avg_waf": waf for i, waf in enumerate(avg_wafs)})
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
            for i, waf in enumerate(avg_wafs):
                print(f"Annotator {i + 1}, Train Avg_waf: {waf:.4f}")
            for i, mae in enumerate(maes):
                print(f"Annotator {i + 1}, Train Avg_mae: {mae:.4f}")
            for i, accuracy in enumerate(accuracys):
                print(f"Annotator {i + 1}, Train Avg_accuracy: {accuracy:.4f}")

            print(f"train_add_frobenius: {add_results['add_frobenius']}")
            print(f"train_add_absolute: {add_results['add_absolute']}")
            print(f"train_add_relative: {add_results['add_relative']}")
            print(f"train_pred_consistency: {add_results['pred_consistency']}")
            print(f"train_true_consistency: {add_results['true_consistency']}")

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

            padded_videos, emotion_labels, valid_masks, original_frames = batch

            padded_videos = padded_videos.cuda(args.local_rank, non_blocking=True)
            emotion_labels = emotion_labels.cuda(args.local_rank, non_blocking=True)
            valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)
            original_frames = original_frames.cuda(args.local_rank, non_blocking=True)

            unpadded_videos = []
            for i in range(len(padded_videos)):
                video = padded_videos[i]
                original_frame_count = original_frames[i].item()
                if original_frame_count < video.size(1):
                    video = video[:, :original_frame_count]
                unpadded_videos.append(video)

            emotion_logits, _ = backbone(unpadded_videos)

            for annotator_id in range(num_annotators):
                valid_samples = valid_masks[:, annotator_id]
                annotator_loss = torch.tensor(0.0, device=valid_samples.device)
                valid_count = 0
                for sample_idx, is_valid in enumerate(valid_samples):
                    if is_valid:
                        sample_logit = emotion_logits[sample_idx, annotator_id, :].unsqueeze(0)
                        sample_target = torch.argmax(emotion_labels[sample_idx, annotator_id]).unsqueeze(0)
                        sample_loss = criterion(sample_logit, sample_target)
                        annotator_loss += sample_loss
                        valid_count += 1
                        with torch.no_grad():
                            _, sample_pred = torch.max(sample_logit, 1)
                            all_preds[annotator_id] = torch.cat([all_preds[annotator_id], sample_pred])
                            all_labels[annotator_id] = torch.cat([all_labels[annotator_id], sample_target])
                    else:
                        all_preds[annotator_id] = torch.cat(
                            [all_preds[annotator_id], torch.tensor([-1], device=all_preds[annotator_id].device)])
                        all_labels[annotator_id] = torch.cat(
                            [all_labels[annotator_id], torch.tensor([-1], device=all_preds[annotator_id].device)])
                if valid_count:
                    annotator_loss /= valid_count
                record_losses[batch_id][annotator_id] += annotator_loss.item()  # xxx X 13

        avg_losses = []
        for annotator_losses in zip(*record_losses):
            valid_losses = [loss for loss in annotator_losses if loss > 0]
            if valid_losses:
                avg_losses.append(sum(valid_losses) / len(valid_losses))
            else:
                avg_losses.append(0.0)
        total_loss = sum(avg_losses)

        avg_wafs = [calculate_waf(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        total_waf = sum(avg_wafs)

        # 计算MAE
        maes = [calculate_mae(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        avg_mae = sum(maes) / len(maes)

        # 计算MAE
        accuracys = [calculate_accuracy(all_preds[a], all_labels[a]) if len(all_labels[a]) > 0 else [] for a in range(num_annotators)]
        avg_accuracy = sum(accuracys) / len(accuracys)

        add_results = add_metric.calculate_add(all_preds, all_labels)

        if args.local_rank == 0:
            whole_mean_kappa, whole_std_kappa, whole_kappa_matrix = calculate_and_plot_kappa(
                all_preds, all_labels, num_annotators, args, "val"
            )
            avg_valid_losses.append(total_loss)

            if not args.evaluate:
                log_dict = {
                    "test_total_loss": total_loss,
                    "test_total_waf": total_waf
                }
                log_dict.update({f"test_annotator_{i + 1}_avg_loss": loss for i, loss in enumerate(avg_losses)})
                log_dict.update({f"test_annotator_{i + 1}_avg_waf": waf for i, waf in enumerate(avg_wafs)})
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
            for i, waf in enumerate(avg_wafs):
                print(f"Annotator {i + 1}, Val Avg_waf: {waf:.4f}")
            for i, mae in enumerate(maes):
                print(f"Annotator {i + 1}, Val Avg_mae: {mae:.4f}")
            for i, accuracy in enumerate(accuracys):
                print(f"Annotator {i + 1}, Val Avg_accuracy: {accuracy:.4f}")

            print(f"avl_add_frobenius: {add_results['add_frobenius']}")
            print(f"avl_add_absolute: {add_results['add_absolute']}")
            print(f"avl_add_relative: {add_results['add_relative']}")
            print(f"avl_pred_consistency: {add_results['pred_consistency']}")
            print(f"avl_true_consistency: {add_results['true_consistency']}")

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

                valid_mask = (preds != -1) & (labels != -1)
                valid_preds = preds[valid_mask]
                valid_labels = labels[valid_mask]

                if len(valid_preds) == 0:
                    kappa_matrix[i, j] = float('nan')
                    continue

                kappa = cohen_kappa_score(valid_preds, valid_labels)
                kappa_matrix[i, j] = kappa

    mask = ~torch.eye(num_annotators, dtype=torch.bool, device=device)
    valid_kappas = kappa_matrix[mask].masked_select(~torch.isnan(kappa_matrix[mask]))
    mean_kappa = valid_kappas.mean().item()
    std_kappa = valid_kappas.std().item()

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

            labels_i = all_labels[i]
            labels_j = all_labels[j]

            valid_mask = (labels_i != -1) & (labels_j != -1)
            valid_labels_i = labels_i[valid_mask]
            valid_labels_j = labels_j[valid_mask]

            if len(valid_labels_i) == 0:
                kappa_matrix[i, j] = float('nan')
                continue

            confusion_matrix = torch.zeros((8, 8), device=device)
            for k in range(len(valid_labels_i)):
                confusion_matrix[valid_labels_i[k].long(), valid_labels_j[k].long()] += 1

            n_samples = len(valid_labels_i)
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
        plt.title(f'Inter-annotator Agreement')
        plt.xlabel('Annotator ID')
        plt.ylabel('Annotator ID')

        plt.savefig(os.path.join(args.save_path, f'{phase}_kappa_heatmap.png'))
        plt.show()
        plt.close()

    mask = ~torch.eye(num_annotators, dtype=torch.bool, device=device)
    valid_kappas = kappa_matrix[mask].masked_select(~torch.isnan(kappa_matrix[mask]))
    mean_kappa = valid_kappas.mean()
    std_kappa = valid_kappas.std()

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
        scheduler = None
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def apply_gradient_clipping(backbone, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=max_norm)


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


class WarmupCosineScheduler:
    def __init__(
            self,
            optimizer,
            max_epoch,
            iters_per_epoch,
            min_lr,
            init_lr,
            warmup_steps=0,
            warmup_start_lr=-1,
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else min_lr
        self.total_steps = max_epoch * iters_per_epoch

        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step

        if total_cur_step < self.warmup_steps:
            progress = float(total_cur_step) / float(max(1, self.warmup_steps))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                lr = self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                group['lr'] = lr
        else:
            progress = float(total_cur_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
                group['lr'] = lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


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
        progress = (step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        lr = (self.init_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + self.min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def warmup_lr_schedule(self, step):
        lr = min(self.init_lr, self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * step / max(self.warmup_steps, 1))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        self.save_path = save_path
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
        save_path = os.path.join(self.save_path, 'checkpoint.pt')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


def main(args):
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        wandb.init(project='MER_Classifier',  name='QFormer')

    train_dataset = VideoDataset(args.video_dir_train, args.csv_file_train)
    train_sampler = DistributedSampler(dataset=train_dataset)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                                  sampler=train_sampler, num_workers=8, collate_fn=dynamic_pad_collate_fn)
    val_dataset = VideoDataset(args.video_dir_val, args.csv_file_val)
    val_sampler = DistributedSampler(dataset=val_dataset)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=args.batch_size, pin_memory=True,
                                sampler=val_sampler, num_workers=8, collate_fn=dynamic_pad_collate_fn)
    if args.evaluate:
        test_dataset = VideoDataset(args.video_dir_test, args.csv_file_test)
        test_sampler = DistributedSampler(dataset=test_dataset)
        dataloader_test = DataLoader(dataset=test_dataset, batch_size=args.batch_size, pin_memory=True,
                                     sampler=test_sampler, num_workers=8, collate_fn=dynamic_pad_collate_fn)

    attention_dim = 256
    feature_dim = 768
    NUM_CATEGORIES = len(EMOTIONS)
    NUM_ANNOTATORS = 13

    backbone = Backbone(vit_model=args.vit_model, q_former_model=args.q_former_model, video_q_former_model=args.video_q_former_model,
                        attention_dim=attention_dim, num_classes=NUM_CATEGORIES, num_annotators=NUM_ANNOTATORS).cuda(local_rank)
    backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(dataloader_train) * args.epochs
    warmup_steps = int(0.2 * total_steps)
    min_lr = 1e-7
    max_lr = 1e-5
    warmup_start_lr = 1e-7

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in backbone.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'lr': max_lr,
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in backbone.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'lr': max_lr * 0.1,
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=max_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        max_epoch=args.epochs,
        iters_per_epoch=len(dataloader_train),
        min_lr=min_lr,
        init_lr=max_lr,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr
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

            video_paths = get_video_paths(args.visualization_save_dir)
            if not video_paths:
                print(f"No videos found in {args.visualization_save_dir}")
                return

            ground_truth_dict = load_ground_truth(args.csv_file_test)

            results_dir = os.path.join(os.path.dirname(args.visualization_save_dir), 'visualization_results_gradcam')

            visualize_specific_videos(
                backbone.module,
                video_paths,
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
        early_stopping = EarlyStopping(save_path=args.save_path, patience=patience, verbose=True, delta=0.001)

    try:
        for epoch in range(initial_epoch, total_epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            avg_train_losses = train_model(backbone, dataloader_train, optimizer, scheduler, criterion, NUM_ANNOTATORS, args, epoch, total_epochs, avg_train_losses)

            avg_valid_losses, valid_loss = evaluate_model(backbone, dataloader_val, criterion, NUM_ANNOTATORS, args, epoch, total_epochs, avg_valid_losses)

            if local_rank == 0:
                early_stopping(valid_loss, backbone.module)
                if early_stopping.early_stop:
                    break

            torch.cuda.empty_cache()

            if (epoch + 1) % args.save_epoch == 0 and local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict_backbone': backbone.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, f'{args.save_path}/checkpoint_epoch_{epoch}.pth')

        if local_rank == 0:
            backbone.module.load_state_dict(torch.load(f'{args.save_path}/checkpoint.pt'))
            torch.save(backbone.module.state_dict(), f'{args.save_path}/checkpoint_best_earlystopping.pth')
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    visualize_specific_videos(backbone, image_paths, results_dir, transform)


def get_video_paths(directory):
    supported_formats = ('.mp4', '.avi', '.mov')
    video_paths = []

    for file in os.listdir(directory):
        if file.lower().endswith(supported_formats):
            video_paths.append(os.path.join(directory, file))

    return sorted(video_paths)


def load_ground_truth(csv_file):
    df = pd.read_csv(csv_file)
    label_dict = {}

    for _, row in df.iterrows():
        video_name = row['name']
        label_dict[video_name] = {}
        label_dict[video_name]['discrete'] = {}
        for i in range(1, 14):
            if f'discrete{i}' in df.columns:
                label_dict[video_name]['discrete'][i - 1] = row[f'discrete{i}']
            else:
                print(f"Warning: Column 'discrete{i}' not found in CSV file")

    return label_dict


def denormalize_frame(frame):
    """反标准化视频帧"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame = frame * std + mean
    frame = frame.clamp(0, 1)
    return frame


def visualize_specific_videos(backbone, video_paths, save_dir, ground_truth_dict):
    backbone.eval()
    device = next(backbone.parameters()).device

    with torch.no_grad():
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            vid_name = os.path.splitext(video_name)[0]
            print(f"Processing video: {video_name}")

            video, _ = load_video_with_resample(
                video_path=video_path,
                n_frms=96,
                height=224,
                width=224,
                return_msg=True
            )
            video = transform(video)
            video = video.unsqueeze(0).to(device)
            num_frames = video.size(2)

            emotion_logits, video_cross_attentions = backbone(video)
            video_cross_attentions = video_cross_attentions[0]

            video_save_dir = os.path.join(save_dir, vid_name)
            os.makedirs(video_save_dir, exist_ok=True)

            for annotator_idx in range(13):
                annotator_dir = os.path.join(video_save_dir, f'annotator_{annotator_idx + 1}')
                os.makedirs(annotator_dir, exist_ok=True)

                curr_attention = video_cross_attentions[:, annotator_idx, :]
                attention_per_frame = []
                for head_idx in range(12):
                    head_att = curr_attention[head_idx]
                    head_att = head_att.reshape(num_frames, -1).mean(dim=1)
                    attention_per_frame.append(head_att)

                attention_weights = torch.stack(attention_per_frame, dim=0)
                attention_weights = torch.mean(attention_weights, dim=0)[0].cpu()
                attention_weights = (attention_weights - attention_weights.min()) / \
                                    (attention_weights.max() - attention_weights.min())

                fig = plt.figure(figsize=(20, 8))
                gs = plt.GridSpec(3, 1, height_ratios=[3, 0.5, 2])

                key_indices = list(range(0, num_frames, 5))
                if (num_frames - 1) not in key_indices:
                    key_indices.append(num_frames - 1)

                frame_grid = gs[1].subgridspec(2, len(key_indices), height_ratios=[2, 0.5], hspace=0.3)

                for i, frame_idx in enumerate(key_indices):
                    frame = video[0, :, frame_idx].cpu()
                    frame = denormalize_frame(frame)
                    frame = frame.permute(1, 2, 0).numpy()

                    ax_frame = fig.add_subplot(frame_grid[0, i])
                    ax_frame.imshow(frame)
                    ax_frame.axis('off')
                    ax_frame.set_title(f'Frame {frame_idx}')

                    ax_att = fig.add_subplot(frame_grid[1, i])
                    att_score = attention_weights[frame_idx].item()
                    ax_att.bar(0, 1, color=plt.cm.Reds(att_score))
                    ax_att.set_xlim(-0.5, 0.5)
                    ax_att.set_ylim(0, 1)
                    ax_att.axis('off')
                    ax_att.text(0, -0.2, f'{att_score:.2f}',
                                ha='center', va='top')

                    frame_fig = plt.figure(figsize=(5, 6))
                    gs_frame = frame_fig.add_gridspec(2, 1, height_ratios=[5, 1])

                    ax_save_frame = frame_fig.add_subplot(gs_frame[0])
                    ax_save_frame.imshow(frame)
                    ax_save_frame.axis('off')
                    ax_save_frame.set_title(f'Frame {frame_idx}')

                    ax_save_att = frame_fig.add_subplot(gs_frame[1])
                    ax_save_att.bar(0, 1, color=plt.cm.Reds(att_score))
                    ax_save_att.set_xlim(-0.5, 0.5)
                    ax_save_att.set_ylim(0, 1)
                    ax_save_att.axis('off')
                    ax_save_att.text(0, -0.2, f'Attention: {att_score:.2f}',
                                     ha='center', va='top')

                    plt.savefig(
                        os.path.join(annotator_dir, f'frame_{frame_idx:03d}.png'),
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1
                    )
                    plt.close(frame_fig)

                pred = torch.argmax(emotion_logits[0, annotator_idx]).item()
                prob = F.softmax(emotion_logits[0, annotator_idx], dim=0)[pred].item()
                pred_emotion = f"{EMOTIONS[pred]} ({prob:.2f})"

                true_emotion = "N/A"
                if vid_name in ground_truth_dict and \
                        'discrete' in ground_truth_dict[vid_name] and \
                        annotator_idx in ground_truth_dict[vid_name]['discrete']:
                    true_label = ground_truth_dict[vid_name]['discrete'][annotator_idx]
                    if pd.notna(true_label):
                        true_emotion = EMOTIONS[int(true_label)]

                title_ax = fig.add_subplot(gs[1])
                title_ax.axis('off')
                title_ax.text(0.5, 4.0,
                              f'Annotator {annotator_idx + 1}\n'
                              f'Prediction: {pred_emotion}, Ground Truth: {true_emotion}',
                              ha='center', va='center')

                ax_curve = fig.add_subplot(gs[2])
                x = np.arange(num_frames)
                plt.plot(x, attention_weights, 'r-', linewidth=2)
                plt.fill_between(x, attention_weights, alpha=0.3, color='red')
                plt.xticks(np.arange(0, num_frames + 1, 5))
                plt.grid(True, axis='x', alpha=0.3)
                plt.xlabel('Frame Index')
                plt.ylabel('Attention Weight')

                plt.tight_layout()
                plt.savefig(os.path.join(annotator_dir, 'attention_visualization.png'),
                            dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()


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


def save_loss_to_file(train_losses, valid_losses, filename='Try_multi_annotation/real_dataset/checkpoints1_class/losses.csv'):
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
    parser.add_argument('--video_dir_train', type=str, default="Try_multi_annotation/real_dataset/video_train", help='video')
    parser.add_argument('--csv_file_train', type=str, default="Try_multi_annotation/real_dataset/train_data.csv", help='label path')
    parser.add_argument('--video_dir_val', type=str, default="Try_multi_annotation/real_dataset/video_val", help='video')
    parser.add_argument('--csv_file_val', type=str, default="Try_multi_annotation/real_dataset/val_data.csv", help='label path')
    parser.add_argument('--video_dir_test', type=str, default="Try_multi_annotation/real_dataset/video_test", help='video')
    parser.add_argument('--csv_file_test', type=str, default="Try_multi_annotation/real_dataset/test_data.csv", help='label path')

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=True)

    parser.add_argument('--vit_model', type=str, default="ckpt/eva-vit-g/eva_vit_g.pth")
    parser.add_argument('--q_former_model', type=str, default="ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument('--video_q_former_model', type=str, default="ckpt/Video-LLaMA-2-7B-Finetuned/VL_LLaMA_2_7B_Finetuned.pth")

    parser.add_argument('--batch_size', type=int, default=8, help="batch_size.")
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
    parser.add_argument('--save_path', type=str, default="Try_multi_annotation/real_dataset/checkpoints1_class", help='')
    parser.add_argument('--save_model', type=str, default="final.pth", help='')

    parser.add_argument('--early_stopped', type=bool, default=True)

    parser.add_argument('--caculate_cross_annotators', type=bool, default=False)

    parser.add_argument('--visualize_annotator_attention', type=bool, default=False)

    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")

    parser.add_argument('--process_directory', type=bool, default=False,
                        help='Whether to process all videos in the directory')
    parser.add_argument('--run_evaluation', type=bool, default=False,
                        help='Whether to run evaluation on test dataset')
    parser.add_argument('--visualization_save_dir', type=str,
                        default='Try_multi_annotation/real_dataset/analysis/specific_videos',
                        help='Directory containing video to process')

    args = parser.parse_args()

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    set_seed(42)

    main(args)