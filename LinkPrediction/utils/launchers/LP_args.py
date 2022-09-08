from typing import NamedTuple
import os 
import argparse 

def get_args(): 
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    args = experiments[int(task_id)] #._asdict()
    print("Launching experiment #: {}, with model: {}, and args:\n{}".format( task_id, args.exp_name, args)) 
    return args
    
class ArgsForSAGE(NamedTuple):
    exp_name: str
    dataset: str
    similarity: str 
    fine_tune: bool  
    top_k: int 
    sigma: float 
    pretrain_epochs:int
    epochs: int
    hidden: int  
    lr: float
    dropout:float
    weight_decay:float 
    layer_neighbors: list
    neg_neighbors: int
    batch_size: int
    n_components: int 


# test_args = ArgsForSAGE(exp_name = 'GS_TEST_JACCARD_BC', dataset= 'BC', pretrain_epochs=0, epochs=2, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[5, 5], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False)

experiments = [
    ArgsForSAGE(exp_name = 'GS_BC_FullRun41_jaccard_neighbortest1', dataset= 'BC', pretrain_epochs=5, epochs=5, hidden=256, lr= 0.001, dropout= 0.001, weight_decay=0.000, layer_neighbors=[5, 5], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, sigma=1.0, similarity='jaccard', fine_tune=False), 
    ArgsForSAGE(exp_name = 'GS_BC_FullRun41_cosine_neighbortest1', dataset= 'BC', pretrain_epochs=5, epochs=5, hidden=256, lr= 0.001, dropout= 0.001, weight_decay=0.000, layer_neighbors=[5, 5], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, sigma=1.0, similarity='cosine', fine_tune=False)] 
    # ArgsForSAGE(exp_name = 'GS_BC_FullRun41_jaccard_neighbortest2', dataset= 'BC', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[10, 10], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False), 
    # ArgsForSAGE(exp_name = 'GS_BC_FullRun41_jaccard_neighbortest3', dataset= 'BC', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[25, 15], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False), 
    # ArgsForSAGE(exp_name = 'GS_BC_FullRun41_jaccard_neighbortest4', dataset= 'BC', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[30, 30], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False), 
    # ArgsForSAGE(exp_name = 'GS_BC_FullRun41_jaccard_neighbortest5', dataset= 'BC', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[35, 35], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False) ]  


    # ArgsForSAGE(exp_name = 'GS_F_FullRun62_jaccard_neighbortest_best_V2', dataset= 'F', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[5, 5], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False), 
    # ArgsForSAGE(exp_name = 'GS_F_FullRun62_jaccard_neighbortest_best_V3', dataset= 'F', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[5, 5], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='jaccard', fine_tune=False), 
    
    # ArgsForSAGE(exp_name = 'GS_F_FullRun62_cosine_neighbortest_best_V2', dataset= 'F', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[10, 10], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='cosine', fine_tune=False), 
    # ArgsForSAGE(exp_name = 'GS_F_FullRun62_cosine_neighbortest_best_V3', dataset= 'F', pretrain_epochs=30, epochs=60, hidden=256, lr= 0.001, dropout= 0.0, weight_decay=0.000, layer_neighbors=[10, 10], neg_neighbors=5, batch_size = 32, n_components=200, top_k=10, similarity='cosine', fine_tune=False), 