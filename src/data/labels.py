
def max_connected_component(G):
    subs = nx.weakly_connected_component_subgraphs(G)
    max(list(subs), lambda H:len(H))

def descendants_out(G, node):
    """  Recursively get all descendents of the DiGraph G,
       descending down the edges, starting at node.""" 

    immediate = [i for _,i in G.out_edges(node)]
    out = set(immediate)

    for node2 in immediate:
        out |= descendants_out(G, node2)
    
    return out


def descendants_in(G, node):
    """  Recursively get all descendents of the DiGraph G,
       ascending up the edges, starting at node.""" 

    immediate = [i for i,_ in G.in_edges(node)]
    out = set(immediate)

    for node2 in immediate:
        out |= descendants_in(G, node2)
    
    return out

def cid_to_terms(G):
    """ Use the term hierarchy G to get the terms annotated to each cid"""
    out = {}
    for term in G.nodes(): 
         try: 
             for cid in G.nodes()[term]['drugs']: 
                 try: 
                     out[cid] |= {term} 
                 except KeyError: 
                     out[cid] = {term} 
         except KeyError: 
             pass
    return out

def filter_n_drug_terms(G, n):
    """ Delete terms in the network if
    annotate less than n drugs."""
    G_copy = G.copy()
    
    
    nodes = G.nodes()
    for node in nodes:
        try:
            if len(G.nodes()[node]['drugs']) < n:
               G_copy.remove_node(node)
        except KeyError:
            pass
    return G_copy


def lowest_in_hierarchy(G, terms):
    """ """
    sub = G.subgraph(terms)
    out = [i for i,v in sub.in_degree() if v == 0]
    return out

import networkx as nx

G_all = nx.read_gml("data/external/All_MeSH_annotations.gml")
G = G_all.subgraph(descendants_in(G_all, "Molecular Mechanisms of Pharmacological Action"))

H = G.copy()
H = filter_n_drug_terms(G, 4)    
H.remove_nodes_from([i for i,_ in G.in_edges('Cytochrome P-450 Enzyme Inhibitors')])
H.remove_node('Cytochrome P-450 Enzyme Inhibitors') # not actually relevant to mechanism
H.remove_node("Enzyme Inhibitors") # way too general
H.remove_node("Neurotransmitter Agents")
H.remove_node('Neurotransmitter Uptake Inhibitors')
#H.remove_node("Membrane Transport Modulator")

import collections
import pandas as pd

# calcualte frequency of different terms
# across all selected drugs
freq = collections.Counter()
cid_to_candidate_terms = {}
for cid, terms in cid_to_terms(H).items():
    temp = lowest_in_hierarchy(H, terms)
    freq.update(temp)
    cid_to_candidate_terms[cid] = temp

freq = pd.Series(freq).sort_values()
def get_most_frequent(terms):
    i = freq.loc[terms].argmax()
    return terms[i]

cid_to_terms = {}
for cid, terms in cid_to_candidate_terms.items():
    cid_to_terms[cid] =  get_most_frequent(terms)
cid_to_terms = pd.Series(cid_to_terms)
cid_to_terms.name = "MeSH"

folders = ["cell_line", "transcriptional", "morphological"]

for folder in folders:
    drug_names = pd.read_csv(f"data/intermediate/{folder}/drug_names.csv", header = None, index_col = 0)
    cid_names = pd.read_csv("data/external/cid_to_names.csv", header = None)
    cid_names.columns = ["NAME","CID"]
    #cid_names.query("NAME in @drug_names")
    cid_names.set_index("NAME", inplace = True)
    cid_names.CID = cid_names.CID.astype('str')
    cid_names = cid_names.join(cid_to_terms, on = 'CID')
    label_df = drug_names.join(cid_names, on = 1)
    
    label_df.to_csv(f"data/intermediate/{folder}/labels.csv")


