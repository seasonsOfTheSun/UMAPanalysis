import numpy as np
import pandas as pd
import re
import sys

import networkx as nx

import urllib.parse as up
import urllib.request as ur

import subprocess as spop
import itertools as it
import json

salt_to_remove = [" maleate", " hydrochloride", " nitrate", 
                  " dihydrochloride", " chloride", " sulfate", 
                  " hydrate", " mesylate", " oxalate", " salt",
                  " from Penicillium brefeldianum", " monohydrate",
                  " trifluoroacetate", " acetate", " isethionate",
                  " hemisulfate", " angular", " sodium", " fumarate",
                  " methanesulfonate", " hemihydrate", " (MW = 374.83)",
                  "\(\+/\-\)-", "\(\+\)-", "\(\-\)-", "S-\(\+\)-", "\(S\)-", "\(±\)-", "D-"]

def remove_salts(cpd):
    try:
        cpd = cpd.lower()
        for s in salt_to_remove:
            cpd = re.sub(s, "", cpd)
        return cpd
    except Exception:
        return np.nan
    
    
def cidFromName(name):
    header = "'Content-Type: application/x-www-form-urlencoded'"
    post_req = 'name='+name
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/cids/txt'

    res = spop.Popen(['curl','--data-urlencode', post_req, url], stderr=spop.PIPE, stdout = spop.PIPE)
    response_string = res.stdout.read().decode('utf-8')

    time.sleep(0.2)
    return response_string.split('\n')[0]

def get_pubchem_info(cid):
    header = "'Content-Type: application/x-www-form-urlencoded'"
    post_req = 'cid='+cid
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/classification/json'
    res = spop.Popen(['curl','--data-urlencode', post_req, url], stdout = spop.PIPE, stderr=spop.PIPE)
    response_string = res.stdout.read().decode('utf-8')
    return json.loads(response_string)

drug_name_file = sys.argv[1]
foldername = sys.argv[2]
drugs = pd.read_csv(drug_name_file, header = None)[0]


name2cid = {}
cid2nodes = {}
G = nx.DiGraph()


for name in drugs:

    tick += 1
    current_time = time.time()
    tps = tick / (current_time - start_time)
    print('Estimated completion at:', time.ctime(current_time + (N - tick) / tps), end = '\r')
    
    try:
        cid = cidFromName(name)
        name2cid[name] = cid
        pchem_info = get_pubchem_info(cid)
        pchem_info = pchem_info['Hierarchies']['Hierarchy']
        for source in pchem_info:
            if source['SourceName'] == 'MeSH':
                drug_classes = source

                temp = []
                for n,i in enumerate(drug_classes['Node']):
                    G.add_node(i['NodeID'], **i['Information'])

                    temp.append(i['NodeID'])
                    for j in i['ParentID']:
                        G.add_edge(i['NodeID'],j)
                        temp.append(j)
                cid2nodes[cid] = temp
    
    except Exception as e:
        print(e)


nx.write_gml(G, f"{foldername}/MeSH_info_hierarchy.gml")

fp = open(f"{foldername}/cid_to_MeSH_info.json", 'w')
json.dump(cid2nodes, fp)
fp.flush()
fp.close()

#
fp = open(f"{foldername}/names_to_cids.json", 'w')
json.dump(name2cid, fp)
fp.flush()
fp.close()
