from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention 

from helper_Methods import *


class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_grams: int,
                 n_kernels: int,
                 conv_out_dim: int):

        super(Conv_KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad = False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad = False).view(1, 1, 1,
                                                                                                          n_kernels)

        # Implement 1 Dimensional CNN layer for each n-gram type
        # Also, use RelU as Activation function
        self.convolutions = []
        for i in range (1, n_grams + 1):
            self.convolutions.append(nn.Sequential(
            nn.ConstantPad1d((0 , i-1 ), 0),
            # the kernel size of the convolutional layer is the same as the current i-gram(uni, bi, tri...) in the loop
            nn.Conv1d(kernel_size = i, in_channels = word_embeddings.get_output_dim(), out_channels = conv_out_dim),
            nn.ReLU()))
            # register conv as part of the model
        self.convolutions = nn.ModuleList(self.convolutions)

        #Cosine similarity matrix
        self.cosine_module = CosineMatrixAttention()


        # Initialize the Linear transformer model:
        # size of the input: number of elements in the soft-TF feautes * number of kernel products (
        # n_kernels *  n_grams * n_grams = all combination of match matrix creation
        # (n-gram pairs from query and document embeddings)
        # the output will be 1 sample
        # also use bias based on the paper formula (by default it's true but just to make sure)
        self.transform = nn.Linear(in_features = n_kernels * n_grams * n_grams, out_features = 1, bias = True)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensorsss
        # -------------------------------------------------------

        # we assume 0 is padding - both need to be removed
        # shape: (batch, query_max)
        #
        query_pad_mask = (query["tokens"] > 0).float()  # > 1 to also mask oov terms
        document_pad_mask = (document["tokens"] > 0).float()

        maskedEmbed = getMaskedEmbed(query_pad_mask, document_pad_mask)
        maskedEmbed = (maskedEmbed.unsqueeze(-1)).cuda()

        #Before the conv
        queryEmbeddings = (self.word_embeddings(query)).cuda()

        documentEmbeddings = (self.word_embeddings(document)).cuda()

        # Transpose the embeddings make it applicible to the convolution layer
        # after the conv feed an relu-layer, it will be transposed back

        query_embeddings_t = queryEmbeddings.transpose(1, 2)
        document_embeddings_t = documentEmbeddings.transpose(1, 2)


        #Initialize list to store each convolutioned n-gram document and query embeddings
        # Do we have to pre-define the sizes of list? can it make the process faster?
        convQueries = []
        convDocs = []

        #Loop through all  n-gram convolution ty
        for conv in self.convolutions:
            # get the embeddings through the layers, and store them in the list in the original row-column format
            convQueries.append(conv(query_embeddings_t).transpose(1, 2))
            convDocs.append(conv(document_embeddings_t).transpose(1, 2))


        #Place sigma and mu into the gpu
        mu = self.mu
        mu = mu.cuda()

        sigma = self.sigma
        sigma = sigma.cuda()

        #Now we have the convolutiend n-gram embeddings for document and queries
        # Next step:
        # For each n-gram combination: create a match matrix: combine each n-gram document and word embeddings:
        # It will provide n*n match matrix
        #Concept: loop through each convolutioned document embedding and calculate the cosine similarity
        # then we have the cosine similarity, apply kernel pooling (where the padding will be masked),
        # then store the results in a list called kernelresult (or softTFFeatures?)
        softTFFeatures = []
        #Initialize the document embedding loop
        for d in convDocs:
            #initialize the inner loop which will provide to loop through all query embeds
            for q in convQueries:
                # Calculate cosine similarity
                matchMatrix = self.cosine_module.forward(q, d)


                #Add a new dimension to resolve mismatch
                matchMatrix = matchMatrix.unsqueeze(-1).cuda()

                # Calculate kernel pooling on the match matrix, input parameters: match matrix and the mask - matrix
                kernelResult = calculateKernel(matchMatrix, maskedEmbed, query_pad_mask, mu = mu, sigma = sigma)
                # the results are the soft-tf features provided by the d-gram document with the q-gram query cosine similarity
                #Store the features in the list
                softTFFeatures.append(kernelResult)


        # Concatenate kernel pooling results/soft-tf features: basicallly it creates a new matrix,
        # where each row is a soft-tf feature (so our list can be considered now as Sequence of tensors),
        # which will be concatenated row-wise?
        pooling_sum = torch.cat(softTFFeatures, 1).cuda()

        # Then Linear transformation will be applied on the matrix
        # The learning - to - rank(LeToR) layer combines the soft-TF ranking features into a ranking score:
        # Steps:
        # apply linear transformation on the concatenated matrixes,
        # calculate hyperbolic tangent on it
        # Create Final Scoring, also Remove the 2nd tensor dimension if it's size is 1
        output = torch.squeeze(torch.tanh(self.transform(pooling_sum)), 1).cuda()
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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
