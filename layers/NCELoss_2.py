# -*- coding: utf-8 -*-
# @Time : 2023/4/8 22:41
# @Author : yysgz
# @File : NCELoss_2.py
# @Project : Re-HAN Model
# @Description :
import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle
import os

class NCELoss(nn.Module):
    """ Class for calculating of the noise-contrasting estimation loss. """
    def __init__(self, opt, vocab_size):
        super(NCELoss, self).__init__()
        # Initialize parameters
        self.vocab_size = vocab_size

    def forward(self, inputs, labels, sampled_values=None):
        """ Performs the forward pass. If sampled_values is None, a log uniform candidate sampler is used
        to obtain the required values. """

        # SHAPES:
        # inputs shape=[batch_size, dims]
        # flat_labels has shape=[batch_size * num_true]
        # sampled_candidates has shape=[num_sampled]
        # true_expected_count has shape=[batch_size, num_true]
        # sampled_expected_count has shape=[num_sampled]
        # all_ids has shape=[batch_size * num_true + num_sampled]
        # true_w has shape=[batch_size * num_true, dims]
        # true_b has shape=[batch_size * num_true]
        # sampled_w has shape=[num_sampled, dims]
        # sampled_b has shape=[num_sampled]
        # row_wise_dots has shape=[batch_size, num_true, dims]
        # dots_as_matrix as size=[batch_size * num_true, dims]
        # true_logits has shape=[batch_size, num_true]
        # sampled_logits has shape=[batch_size, num_sampled]

        flat_labels = labels.view([-1])
        num_true = labels.size()[1]
        true_per_batch = flat_labels.size()[0]

        # Stop gradients for the sampled values
        sampled_candidates, true_expected_count, sampled_expected_count = (s.detach() for s in sampled_values)

        print('Calculating the NCE loss ...')
        # Concatenate true and sampled labels
        all_ids = torch.cat((flat_labels, sampled_candidates), 0)
        # Look up the embeddings of the combined labels


        # Obtain true logits
        tw_c = true_w.size()[1]
        true_w = true_w.view(-1, num_true, tw_c)
        row_wise_dots = inputs.unsqueeze(1) * true_w
        dots_as_matrix = row_wise_dots.view(-1, tw_c)
        true_logits = torch.sum(dots_as_matrix, 1).view(-1, num_true)
        true_b = true_b.view(-1, num_true)
        true_logits += true_b.expand_as(true_logits)
        # Obtain sampled logits; @ is the matmul operator
        sampled_logits = inputs @ sampled_w.t()
        sampled_logits += sampled_b.expand_as(sampled_logits)

        if self.opt.subtract_log_q:
            print('Subtracting log(Q(y|x)) ...')
            # Subtract the log expected count of the labels in the sample to get the logits of the true labels
            true_logits -= torch.log(true_expected_count)
            sampled_logits -= torch.log(sampled_expected_count.expand_as(sampled_logits))

        # Construct output logits and labels
        out_logits = torch.cat((true_logits, sampled_logits), 1)
        # Divide true logit labels by num_true to ensure the per-example labels sum to 1.0,
        # i.e. form a proper probability distribution.
        true_logit_labels = torch.ones(true_logits.size()) / num_true
        sampled_logit_labels = torch.zeros(sampled_logits.size())
        out_labels = torch.cat((true_logit_labels, sampled_logit_labels), 1)
        out_labels = Variable(out_labels)

        # Calculate the sampled losses (equivalent to TFs 'sigmoid_cross_entropy_with_logits')
        loss_criterion = nn.BCELoss()
        nce_loss = loss_criterion(torch.sigmoid(out_logits), out_labels)
        return nce_loss