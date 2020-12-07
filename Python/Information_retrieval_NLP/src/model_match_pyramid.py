from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict
import torch
import torch.nn as nn                            
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention


class MatchPyramid(nn.Module):
    '''
    Paper: Text Matching as Image Recognition, Pang et al., AAAI'16
    '''

    def __init__(self,
                 #The embedding layer is specified as an AllenNLP TextFieldEmbedder.
                 word_embeddings: TextFieldEmbedder,
                 #the size of output channels
                 conv_output_size: List[int],
                 #the size of input channels
                 conv_kernel_size: List[Tuple[int,int]],
                 # the size of pooling layers to reduce the dimension of the feature maps
                 adaptive_pooling_size: List[Tuple[int,int]]):

        super(MatchPyramid, self).__init__()

        self.word_embeddings = word_embeddings
        self.cosine_module = CosineMatrixAttention()

        if len(conv_output_size) != len(conv_kernel_size) or len(conv_output_size) != len(adaptive_pooling_size):
            raise Exception("conv_output_size, conv_kernel_size, adaptive_pooling_size must have the same length")

        #define the dictionary of convolution layers 
        conv_layer_dict = OrderedDict()
        last_channel_out = 1
        for i in range(len(conv_output_size)):
            #pads the input tensor boundaries with a constant value
            #padding((padding_left, padding_right,padding_bottom),tuple)
            conv_layer_dict["pad " +str(i)] = nn.ConstantPad2d((0,conv_kernel_size[i][0] - 1,0, conv_kernel_size[i][1] - 1), 0)
            #applies a 2D convolution 
            conv_layer_dict["conv "+str(i)] = nn.Conv2d(kernel_size=conv_kernel_size[i], in_channels=last_channel_out, out_channels=conv_output_size[i])
            #applies a ReLU activation function
            conv_layer_dict["relu "+str(i)] = nn.ReLU()
            #applies a 2D adaptive max pooling 
            conv_layer_dict["pool "+str(i)] = nn.AdaptiveMaxPool2d(adaptive_pooling_size[i]) 
            
            last_channel_out = conv_output_size[i]
            
        #add the layers to the model
        self.conv_layers = nn.Sequential(conv_layer_dict)
        
        ##adding FC layers
        self.dense = nn.Linear(conv_output_size[-1] * adaptive_pooling_size[-1][0] * adaptive_pooling_size[-1][1], out_features=100, bias=True)
        self.dense2 = nn.Linear(100, out_features=10, bias=True)
        self.dense3 = nn.Linear(10, out_features=1, bias=False)
        
        #initialize weights (values are taken from matchzoo)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)
        
        #initialize biases
        self.dense.bias.data.fill_(0.0)
        

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        #similarity matrix
        #shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix = cosine_matrix[:,None,:,:]
        
        #convolution
        #shape: (batch, conv_output_size, query_max, doc_max) 
        conv_result = self.conv_layers(cosine_matrix)
        
        #dynamic pooling
        #flatten the output of dynamic pooling
        #shape: (batch, conv_output_size * pool_h * pool_w) 
        conv_result_flat = conv_result.view(conv_result.size(0), -1)

        #
        # Learning to rank layer
        # -------------------------------------------------------
        dense_out = F.relu(self.dense(conv_result_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)
        output = torch.squeeze(dense_out, 1)
        return output
             