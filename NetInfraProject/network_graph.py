#!/usr/bin/python3

# la commande de conversion vers le graph est :
#  dot -Tpng graph.dot -o graph.png

import subprocess,re,functools

nom_graphe = "Network Infrastructure Project (L2TPv3 LAB)"

interface_desc = """        "%s" [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="lightblue"><b>%s</b></td></tr>
                <tr><td align="left">IP: %s</td></tr>
            </table>
        >];"""
switch_desc = """        "%s" [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="orange"><b>%s</b></td></tr>
            </table>
        >];"""

nom_fichier = 'build_architecture.sh'
contenu_fichier = open(nom_fichier).readlines()

list_hosts = [t.split()[3] for t in contenu_fichier if re.findall("ip\s+netns\s+add",t)]
list_switches = [t.split()[2] for t in contenu_fichier if re.findall("ovs-vsctl\s+add-br",t)]
list_veths = [[t.split()[8],t.split()[3]] for t in contenu_fichier if re.findall("ip\s+link\s+add",t)]
list_ports = [[t.split()[2],t.split()[3]] for t in contenu_fichier if re.findall("ovs-vsctl\s+add-port",t)]
list_addresses = [[t.split()[8],t.split()[9]] for t in contenu_fichier if re.findall("ip\s+a(ddress)?\s+add\s+dev",t)]

list_veths = [sorted(x,key=functools.cmp_to_key(lambda x,y: -1 if (x[:x.find('-')] in list_switches) else 1))  for x in list_veths]

print ("""digraph G { 
    label = "%s";
    labelloc = top;
 
    node [shape=record];
    edge [dir=both];"""%(nom_graphe))

compteur = 0
for h in list_hosts:
	print ("""subgraph cluster_%d {
		label = %s;
	"""%(compteur,h))
	compteur+=1
	interfaces = [i for i in list_addresses if i[0].startswith(h)]
	for i in interfaces:
		print (interface_desc%(i[0],i[0],i[1]))
	print ("""	}""")	

for s in list_switches:
	print ("""subgraph cluster_%d {
		label = %s;
	"""%(compteur,s))
	compteur+=1
	print (switch_desc%(s,s))
	print ("""	}""")

dico_veths = dict(list_veths)
for p in list_ports:
	print (""" "%s"->"%s"; """%(p[0],dico_veths[p[1]]))
print ("}")
