
import torch



# Calculate kernel on knrm and conv knrm
def calculateKernel(matchMatrix, maskedMatrix,queryMask,mu,sigma):
    #Calculate RBF - Kernel
    rbfKernel = torch.exp(- torch.pow(matchMatrix - mu, 2) / (2 * torch.pow(sigma, 2))).cuda()
    #Mask out the padding from RBF-Kernel
    # The kernel models need masking after the kernels -> the padding 0's will become non-zero,
    # because of the kernel (but when summed up again will distort the output) --> The masking has to be done before sum up
    #Filter out the paddings from the RBF-Kernels
    rbfKernel_Masked = (rbfKernel * maskedMatrix).cuda()

    # Sum up the RBF-Kernel over the document dimension
    rbfKernel_sum = torch.sum(rbfKernel_Masked, 2).cuda()
    #Calculate the logarithm of the summed kernels

    # set the negative values to 1e-10, else we will have a lot nan-s: in the paper-related github (and also other sources):
    # in case of tensorflow, 'tf.maximum(<summed RBF kernel> , 1e-10)' were used, which also only modify the negative context values
    # Also, by multiplicating with  0.01 will be used to scale down the data
    rbfLog = torch.log(torch.clamp(rbfKernel_sum, min=1e-10)) * 0.01

    # Mask out the paddings over the query dimension
    rbfLogMasked = (rbfLog * queryMask.unsqueeze(-1)).cuda()

    # Create soft-TF features: sum the log-values over the query dimension
    softTFFeat = torch.sum(rbfLogMasked, 1).cuda()
    #print(softTFFeat)

    return softTFFeat
def getMaskedEmbed(query_pad_filtered, document_pad_filtered):
    #The  kernel models need masking after the kernels -> the padding 0
    #'s will become non-zero, because of the kernel (but when summed up again will distort the output)

    # Get the dot product of the unmasked queries --> the RBF kernels will be multiplicated
    # with that, which will provide a lower sized (since the padding will be filtered out) matrix
    # flatten the pad-filtered queries, and documents, then transpose the documents and divide it to two columns:
    # Then apply matrix multiplication (batch matrix-matrix product of matrices?)
    return torch.bmm(query_pad_filtered.unsqueeze(-1), document_pad_filtered.unsqueeze(-1).transpose(-1, -2)).cuda()


#Results saver method from TUWEL
def save_sorted_results(results, file, until_rank=-1):
    with open(file, "w") as val_file:
        lines = 0
        for query_id, query_data in results.items():

            # sort the results per query based on the output
            for rank_i, (doc_id, output_value) in enumerate(sorted(query_data, key=lambda x: x[1], reverse=True)):
                val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value]) + "\n")
                lines += 1
                if until_rank > -1 and rank_i == until_rank + 1:
                    break
    return lines



#https://github.com/sebastian-hofstaetter/sigir19-neural-ir/blob/master/matchmaker/performance_monitor.py?fbclid=IwAR3qpQOFdGwSaJIUaXgXIQ4pwlNtVtsirLwvPU5gvRTqMfBU2zlOAeSaDTM
from timeit import default_timer

class PerformanceMonitor():
    def __init__(self):
        self.timings = {}
        self.current_times = {}

    def start_block(self,category:str):
        self.current_times[category] = default_timer()

    def stop_block(self,category:str,instances:int=1):
        if not category in self.timings:
            self.timings[category] = (0,0)

        time,old_instances = self.timings[category]

        self.timings[category] = (time + default_timer() - self.current_times[category], old_instances + instances)

    def print_summary(self):
        for cat,(time,instances) in self.timings.items():
            if instances > 1:
                print(cat, instances/time, "it/s")
            else:
                print(cat, time, "s")

    def save_summary(self, file):
        with open(file, "w") as out_file:
            for cat,(time,instances) in self.timings.items():
                if instances > 1:
                    out_file.write(cat + str(instances/time) + "it/s\n")
                else:
                    out_file.write(cat + str(time) + "s\n")

