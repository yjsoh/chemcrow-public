import click

from chemcrow.agents import ChemCrow; 
from datasets import load_dataset

runs = load_dataset("csv", data_files="chemcrow_tasks.csv")

chem_model = ChemCrow(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tools_model="meta-llama/Meta-Llama-3.1-8B-Instruct", temp=0.1)

@click.command()
@click.option('--filename', prompt="name of output file", help='name of output file')

def main(filename):
    with open(filename, "a", encoding="utf-8") as file:
        for run in runs["train"]:
            file.write(run['src'] + " " + run['type'])
            file.write(run['prompt'])
            if run['type'] != 'locked':
                success = False
                out = ""
                while not success:
                    try:
                        out = chem_model.run(run['prompt'])
                        success = True
                    except:
                        success = False
                file.write(out)
            else:
                file.write("Task locked, moving onto next task...\n")

main()
