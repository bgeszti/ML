import sys

sys.path.append('/content/drive/My Drive/air/src')

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment

prepare_environment(Params({}))  # sets the seeds to be fixed

import torch
import allennlp.data
from allennlp.data.iterators import BucketIterator
from allennlp.data import *
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from data_loading import *
from model_knrm import *
from model_conv_knrm import *
from model_match_pyramid import *

# change paths to your data directory
# executing path is /content -> so change paths accordingly
pathPrefix = "/content/drive/My Drive"
config = {
    "vocab_directory": pathPrefix + "/data/allen_vocab_lower_10",
    "pre_trained_embedding": pathPrefix + "/data/glove.42B.300d.txt",
    "model": "knrm",
    # "model": "conv_knrm",
    "train_data": pathPrefix + "/data/triples.train.tsv",
    "validation_data": pathPrefix + "/data/msmarco_tuples.validation.tsv",
    "test_data": pathPrefix + "/data/msmarco_tuples.test.tsv",
}

#
# data loading
#

vocab = Vocabulary.from_files(config["vocab_directory"])
tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                       "embedding_dim": 300,
                                                       "trainable": True,
                                                       "padding_index": 0}))

word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    model = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "match_pyramid":
    model = MatchPyramid(word_embedder, conv_output_size=[16, 16, 16, 16, 16],
                         conv_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                         adaptive_pooling_size=[[36, 90], [18, 60], [9, 30], [6, 20], [3, 10]])

# todo optimizer, loss

print('Model', config["model"], 'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#
iterBatchSize = 64
_triple_loader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30, tokenizer=WordTokenizer(
    word_splitter=JustSpacesWordSplitter()))  # already spacy tokenized, so that it is faster

_iterator = BucketIterator(batch_size=iterBatchSize,
                           sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

_iterator.index_with(vocab)

# Create a folder which will store the model state, and the results: model name + current timestamp without seconds
from datetime import datetime
import os

dt_string = datetime.now().strftime("%d-%m-%Y-%H_%M")
newFolder = str(config["model"]) + "_" + dt_string + '/'
resultFolder = pathPrefix + '/air_results/' + newFolder
os.mkdir(resultFolder)

# %%

from helper_Methods import *
from core_metrics import *
from allennlp.nn.util import *

# to hide "elementwise-mean is deprecated" warning
loss_functionCriterion = torch.nn.MarginRankingLoss(margin=1, reduction='mean')
# Initialize ADAM optimizer
optimAdam = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# TODO: check that how can the batch size can retrieved from the class
trainBatchSize = iterBatchSize
# initiailize early stopping: We will decide based on the MRR that the model is better or not
valMetricRes = {}
valMetricRes["MRR@10"] = 0
earlyStop = False

# creat lists to monitor loss
train_losses = {}
perf_monitor = PerformanceMonitor()
monitorModel = "train-" + str(config["model"])

# creat lists to store the validation results for vizualization
allValResults = {}

iterCounter = 0
for epoch in range(2):
    # if early stopping has been triggered during validation, exit from the epoch
    if earlyStop is True:
        break

    perf_monitor.start_block(monitorModel)
    perf_start_inst = 0
    # prep model for training
    model.train()
    # Creating a label tensor filled with ones --> will be needed for marginranking loss
    # should be initialized in each outer loop, since in the last loop in the inner cycle the size of the tensor
    # will probably change
    label = torch.ones(trainBatchSize).cuda()
    batchCounter = 0
    # Train loop
    for batch in Tqdm.tqdm(_iterator(_triple_loader.read(config["train_data"]), num_epochs=1)):
        iterCounter += 1
        batch = move_to_device(batch, 0)
        # batch  = Parameter(batch).to(device)
        batchCounter += 1
        model.train()
        # todo train loop
        # in the beggining of each train loop, clean the optimizer (zero_grad() method)
        optimAdam.zero_grad()
        # retrieve the current batch size --> The iterators do not guarantee a fixed batch size
        # (the last one will probably be smaller) --> so we will retrieve the number of tokens from e.g. the query
        currentBatchSize = batch["query_tokens"]["tokens"].shape[0]
        # for the batch size, th

        # based on the slides, the model will be trained with triplets:
        # Triple: 1 query, 1 relevant + 1 non relevant document
        # where the relevant documents are: batch['doc_pos_tokens']
        # and the non relevants: batch['doc_neg_tokens']
        # the goal is to maximize the margin between these documents, that's why we apply the margin loss
        # we will apply the 2 'forward' pass:
        # so we will call model.forward() twice, once for the relevant, and second time for the non-relevant doc
        relevantDocsOutPut = model.forward(batch["query_tokens"], batch["doc_pos_tokens"]).cuda()

        nonRelevantDocsOutPut = model.forward(batch["query_tokens"], batch["doc_neg_tokens"]).cuda()

        # If the last batch is smallert than the other ones, we need a smaller label tensors, else we will get
        # dimensionality mismatch exceptions
        if currentBatchSize != trainBatchSize:
            # reduce the label tensor size to the size of the last (or current batch)
            label = torch.ones(currentBatchSize).cuda()

        # calculate the loss --> it's important to place the relevant docs as the first parameter, due to the usage
        # of the loss function: If y (the label) = 1 then it assumed the first input should be ranked higher
        # (have a larger value) than the second input,  # and vice-versa for y = -1
        loss = loss_functionCriterion(relevantDocsOutPut, nonRelevantDocsOutPut, label).cuda(torch.device("cuda"))

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        # optimizer step() will calculate the loss
        optimAdam.step()
        # record training loss
        # train_losses.append(loss.item())

        train_losses.setdefault(iterCounter, loss.item())

        label = torch.ones(trainBatchSize).cuda()

        # If the  batchCOunter divisor with 7500 returns 0: evaluate the model against the validation set
        if batchCounter % 7500 == 0:
            _tuple_loader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180,
                                                        max_query_length=30)  # not spacy tokenized already (default is spacy)
            _iterValidation = BucketIterator(batch_size=128,
                                             sorting_keys=[("doc_tokens", "num_tokens"),
                                                           ("query_tokens", "num_tokens")])
            _iterValidation.index_with(vocab)
            # Validation loop
            # Set the model to evaluation
            validationResults = {}
            model.eval()

            perf_monitor.stop_block(monitorModel, (batchCounter - perf_start_inst) * currentBatchSize)
            perf_start_inst = batchCounter
            perfVal = "validation-" + str(config["model"])
            perf_monitor.start_block(perfVal)

            totalSize = 0
            for batchVal in Tqdm.tqdm(_iterValidation(_tuple_loader.read(config["validation_data"]), num_epochs=1)):
                totalSize += currentBatchSize
                batchVal = move_to_device(batchVal, 0)

                currentBatchSize = batchVal["query_tokens"]["tokens"].shape[0]
                output = model.forward(batchVal["query_tokens"], batchVal["doc_tokens"]).cuda()
                for q, d, o in zip(batchVal["query_id"].tolist(), batchVal["doc_id"].tolist(), output.tolist()):
                    queryID = str(q)
                    docID = str(d)
                    if queryID not in validationResults:
                        validationResults[queryID] = []
                    # print('+++++++++++++++++++++++++++++++++++++++++')
                    # print("qr: " + str(q) + " docid: " + str(d) + " output: " + str(o))
                    validationResults[queryID].append((docID, float(o)))

            perf_monitor.stop_block(perfVal, totalSize)
            perf_monitor.start_block(monitorModel)
            perf_monitor.print_summary()
            ranked_validadtionResults = unrolled_to_ranked_result(validationResults)

            # Create the validation results to ranked results:
            metrics = calculate_metrics_plain(ranked_validadtionResults,
                                              load_qrels(pathPrefix + "/data/msmarco_qrels.txt"),
                                              binarization_point=1)

            if iterCounter not in allValResults:
                allValResults[iterCounter] = {}
            # Store the results
            allValResults[iterCounter] = metrics
            # allValResults.append(metrics)

            print('#####################')
            for metric in metrics:
                print('{}: {}'.format(metric, metrics[metric]))
            print('#####################')
            print('==========================')
            # print(metrics.keys())

            # If the current MRR is better than the previous one, overwrite it, and save the model maybe?
            print("actual MRR: " + str(metrics["MRR@10"]))
            print("reference MRR : " + str(valMetricRes["MRR@10"]))
            print('==========================')

            if (metrics["MRR@10"] > valMetricRes["MRR@10"]):
                valMetricRes["MRR@10"] = metrics["MRR@10"]
                # Save the model
                torch.save(model.state_dict(), resultFolder + str(config["model"]) + "_best-model.pytorch-state-dict")
                modelToSave = model.state_dict()

            elif (metrics["MRR@10"] <= valMetricRes["MRR@10"]):
                earlyStop = True
                print("Early stopping initiated")
                break

    perf_monitor.stop_block(monitorModel, batchCounter - perf_start_inst)

# %%

# Load the model from the best state

_tuple_loader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180,
                                            max_query_length=30)  # not spacy tokenized already (default is spacy)
_iterator = BucketIterator(batch_size=128,
                           sorting_keys=[("doc_tokens", "num_tokens"), ("query_tokens", "num_tokens")])
_iterator.index_with(vocab)

# create an evaluation model, then load back the model state with the best validation result
if config["model"] == "knrm":
    modelToEval = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "conv_knrm":
    modelToEval = Conv_KNRM(word_embedder, n_grams=3, n_kernels=11, conv_out_dim=128)
elif config["model"] == "match_pyramid":
    modelToEval = MatchPyramid(word_embedder, conv_output_size=[16, 16, 16, 16, 16],
                               conv_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                               adaptive_pooling_size=[[36, 90], [18, 60], [9, 30], [6, 20], [3, 10]])

modelToEval.load_state_dict(modelToSave)
modelToEval = modelToEval.to(device)

# %%

# Evaluate the results on MSMARCO
perfEvalMsMarco = "test-msmarcoPerformance" + str(config["model"])
perf_monitor.start_block(perfEvalMsMarco)
modelToEval.eval()
msMarcoresults = {}
for batch in Tqdm.tqdm(_iterator(_tuple_loader.read(config["test_data"]), num_epochs=1)):
    # todo test loop
    batch = move_to_device(batch, 0)
    # output = model.forward(batch["query_tokens"], batch["doc_tokens"]).cuda()
    output = modelToEval.forward(batch["query_tokens"], batch["doc_tokens"]).cuda()
    for q, d, o in zip(batch["query_id"].tolist(), batch["doc_id"].tolist(), output.tolist()):
        queryID = str(q)
        docID = str(d)
        if queryID not in msMarcoresults:
            msMarcoresults[queryID] = []
        # print('+++++++++++++++++++++++++++++++++++++++++')
        # print("qr: " + str(q) + " docid: " + str(d) + " output: " + str(o))
        msMarcoresults[queryID].append((docID, float(o)))

perf_monitor.stop_block(perfEvalMsMarco)
# pass
# Set the results to appropriate format


# %%

msMarcoRankedResults = unrolled_to_ranked_result(msMarcoresults)

# %%

msMarcoevaluationResults = calculate_metrics_plain(msMarcoRankedResults,
                                                   load_qrels(pathPrefix + "/data/msmarco_qrels.txt"),
                                                   binarization_point=1)

# %%

print('#####################')
print('eval on ms marco test set')
for metric in msMarcoevaluationResults:
    print('{}: {}'.format(metric, msMarcoevaluationResults[metric]))
print('#####################')
print('==========================')
# print(metrics.keys())
print('==========================')

# %%

# Evaluate the results on the FIRA dataset
perfEvalFira = "test-firaPerformance" + str(config["model"])
perf_monitor.start_block(perfEvalFira)

modelToEval.eval()
firaResults = {}
for batch in Tqdm.tqdm(
        _iterator(_tuple_loader.read(pathPrefix + "/data/fira_numsnippets_test_tuples.tsv"), num_epochs=1)):
    # todo test loop
    batch = move_to_device(batch, 0)
    # output = model.forward(batch["query_tokens"], batch["doc_tokens"]).cuda()
    output = modelToEval.forward(batch["query_tokens"], batch["doc_tokens"]).cuda()
    for q, d, o in zip(batch["query_id"].tolist(), batch["doc_id"].tolist(), output.tolist()):
        queryID = str(q)
        docID = str(d)
        if queryID not in firaResults:
            firaResults[queryID] = []

        firaResults[queryID].append((docID, float(o)))

perf_monitor.stop_block(perfEvalFira)

# %%

# Set the results to appropriate format
firaRankedResults = unrolled_to_ranked_result(firaResults)

firaevaluationResults = calculate_metrics_plain(firaRankedResults,
                                                load_qrels(pathPrefix + "/data/fira_numsnippets_qrels.txt"),
                                                binarization_point=1)

# %%

print('#####################')
print('eval on fira test set')
for metric in firaevaluationResults:
    print('{}: {}'.format(metric, firaevaluationResults[metric]))
print('#####################')
print('==========================')
# print(metrics.keys())
print('==========================')

# %%

# !pip install pandas
import pandas as pd

res = pd.concat({k: pd.DataFrame(v) for k, v in firaResults.items()}, axis=0)
result = pd.DataFrame(res.values, index=res.index.droplevel(1), columns=['document', 'relavance']).to_csv(
    resultFolder + str(config["model"]) + '_firaRelevances.csv')

# %%


# to pandas

dfMarco = pd.DataFrame.from_dict(msMarcoevaluationResults, orient="index").to_csv(
    resultFolder + str(config["model"]) + '_msmarco.csv')

fr = pd.DataFrame.from_dict(firaevaluationResults, orient="index").to_csv(
    resultFolder + str(config["model"]) + '_fira.csv')

# %%

allValidationResultsPd = pd.DataFrame.from_dict({(i): allValResults[i] for i in allValResults.keys()},
                                                orient='index')

# %%

allValidationResultsPd.to_csv(resultFolder + str(config["model"]) + '_validationResults.csv')

# %%

tl = pd.DataFrame.from_dict(train_losses, orient="index")

# %%

tl.to_csv(resultFolder + str(config["model"]) + '_lossResults.csv')
