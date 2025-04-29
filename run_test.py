from chemcrow.agents import ChemCrow; 
from datasets import load_dataset

runs = load_dataset("csv", data_files="chemcrow_tasks.csv")

chem_model = ChemCrow(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tools_model="meta-llama/Meta-Llama-3.1-8B-Instruct", temp=0.1)

for run in runs["train"]:
    print(run['src'] + " " + run['type'])
    print(run['prompt'])
    if run['type'] != 'locked':
        success = False
        out = ""
        while not success:
            try:
                out = chem_model.run(run['prompt'])
                success = True
            except:
                success = False
        print(out)
    else:
        print("Task locked, moving onto next task...\n")

