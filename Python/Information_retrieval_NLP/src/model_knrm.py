from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder

from helper_Methods import *
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention 

class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        #Cosine matrix
        self.cosine_module = CosineMatrixAttention()
        # Initialize the Linear transformer model:
        self.transform = nn.Linear(n_kernels, out_features = 1, bias = True)


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float().cuda() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float().cuda()


        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)
        
        #Create a mask matrix
        maskedEmbed = getMaskedEmbed(query_pad_oov_mask, document_pad_oov_mask)
        maskedEmbed = maskedEmbed.unsqueeze(-1).cuda()
        #
        # cosine matrix
        # -------------------------------------------------------
        matchMatrix = self.cosine_module.forward(query_embeddings, document_embeddings)

        # Add an extra dimension the solve the dimensionality mismatch
        matchMatrix = matchMatrix.unsqueeze(-1).cuda()

        mu = self.mu
        mu = mu.cuda()

        sigma = self.sigma
        sigma = sigma.cuda()

        #Calculate the Soft-TF features from the Matchmatrix
        sofTFFeatures = calculateKernel(matchMatrix = matchMatrix, maskedMatrix = maskedEmbed, queryMask = query_pad_oov_mask,
                                        mu = mu, sigma = sigma)

        # apply linear transformation on the soft tf features,
        # calculate hyperbolic tangent on it
        # Remove the 2nd tensor dimension if it's size is 1
        output = torch.squeeze(torch.tanh(self.transform(sofTFFeatures)), 1).cuda()
        
        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
