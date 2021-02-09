import sys
import networkx as nx


path  = sys.argv[1]
m = re.match("*/networks/(?P<name>.*?).gml", path)
name = m.groupdict()['name']

G = nx.read_gml(path)

layouts = nx.spring_layout(G)
