#!/usr/bin/env python3
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tools_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temp=0.1,
)
chem_model.run("What is the volatility of butadiene?")
