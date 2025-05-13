#!/usr/bin/env python3
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tools_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temp=0.1,
)

chem_model.run("Tell me what the boiling point is of the reaction product between isoamyl acetate and ethanol. To do this, predict the product of this reaction, and find its boiling point in celsius using cas number.")

