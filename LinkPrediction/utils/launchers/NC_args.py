
from typing import NamedTuple

class ArgsForSAGE(NamedTuple):
    exp_name: str
    dataset: str 
    top_k: int 
    fine_tune: bool 
    similarity: str
    pretrain_epochs:int
    epochs: int
    hidden: int  
    dropout:float
    lr: float
    weight_decay:float 
    layer_neighbors: list
    batch_size: int
    n_components: int 


test_args = ArgsForSAGE(exp_name='TEST', dataset='Cora', top_k=10, fine_tune=False, similarity='cosine',
    pretrain_epochs = 60, epochs = 30, hidden = 256, dropout=0.0, lr = 0.001, weight_decay= 0.001, layer_neighbors=[15,15],  batch_size = 32, n_components=200)