import os 
from datetime import date 
from torch.nn.functional import cosine_similarity

def save_results(exp_name, args, res, wandb_configs=None): 
    path = '/home/mila/r/rebecca.salganik/Projects/Fairness_locality/logs/EXPERIMENTS/'
    # if not os.path.exists(path): 
    #     os.mkdir(path)
    today = date.today().strftime("%d-%m") 
    save_name = path+'{}--{}.txt'.format(today, exp_name)
    print("***Saving results to:{}***".format(save_name))
    with open(save_name, "w") as f: 
        f.write("***ARGS***\n")
        for k in args._asdict().keys(): 
            f.write("{} : {}\n".format(k, args._asdict()[k]))
        if wandb_configs:
            f.write("***WANDB_INFO***\n")
            f.write("{} : {}\n".format("ID", wandb_configs.id))
            f.write("{} : {}\n".format("NAME", wandb_configs.name))
        f.write("***RESULTS***\n")
        for k in res.keys(): 
            f.write("{} : {}\n".format(k, res[k]))
        
def compare_embeddings(utility_emb, fair_emb, metric='cosine'): 
    return cosine_similarity(utility_emb, fair_emb) 