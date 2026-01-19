import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np

path = "./topology/"
if not os.path.exists(path):
    os.makedirs(path)

np.random.seed(10)

# SAGIN
node_s = 1
node_a = 6
node_g = 18
max_node = node_s + node_a + node_g

s_g_dis = 500 #km
a_g_dis = 10 #km
s_a_dis = s_g_dis - a_g_dis

a_a_dis = 10 #km
g_g_dis = 7.8375
g_g_dis_grougp = 5 #km

def generate_ground_graph(MAX_NODE, p, seed):
    # path = AI_model.path # "D:/元智資工/專題製作/Code/DDPG/DDPG(our)/test/DDPG(mig_placement_AI)/new/VNF/Image/test/"
    link_num = 0

    MEC_Node_Graph = []
    MEC_Link_Graph = []
    Link_propagation = []

    G = nx.erdos_renyi_graph(MAX_NODE, p, seed = seed)
    pos = nx.spring_layout(G, seed = seed)

    n_labels = {}
    e_labels = {}
    e_i_labels = {}

    for i in range(MAX_NODE):
        n_labels[i] = str(i)
        key_lst = [k for k in G.adj[i]]
        MEC_Node_Graph.append(sorted(key_lst))

        # MEC_Link_Graph.append()
        for j in key_lst:
            if e_labels.get((j, i), None) == None:
                e_labels[(i, j)] = round((MEC_min_dis + (MEC_max_dis - MEC_min_dis) * random.random()) *  ms_per_km, 3) # 0.025~0.03 ms
                # print("e_labels[(i, j)] = ",e_labels[(i, j)])
                Link_propagation.append(e_labels[(i, j)])
                e_i_labels[(i, j)] = link_num # link index
                link_num += 1
        
        temp_lst = []
        for j in MEC_Node_Graph[i]:
            if e_i_labels.get((i, j), None) != None:
                temp_lst.append(e_i_labels[(i, j)])
            elif e_i_labels.get((j, i), None) != None:
                temp_lst.append(e_i_labels[(j, i)])
        
        MEC_Link_Graph.append(temp_lst)

    name = "Network_weighted_graph"
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_labels(G, pos, n_labels, font_size=20, font_color="whitesmoke", font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, e_labels, font_size=10, verticalalignment = "bottom", rotate=False)
    
 
    nx.draw(G, pos, node_size=2000)
    # adjust_text(e_labels, )
    figure = plt.gcf()
    figure.savefig(path + name)

    name = "Network_index_graph"
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_labels(G, pos, n_labels, font_size=20, font_color="whitesmoke", font_weight="bold")
    # nx.draw_networkx_edge_labels(G, pos, e_i_labels, font_size=12, verticalalignment = "bottom", rotate=False)
    nx.draw(G, pos, node_size=2000)

    figure = plt.gcf()
    figure.savefig(path + name)

    return MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num

def generate_graph():
    # const
    ms_per_km = 0.0033 # ms
    # ms_per_km = 1/ 299792/1000 # ms

    # seed = 38
    # random.seed(seed)

    # node #0:satellite #1~3:UAV #4~8:BS
    node = [x for x in range(max_node)]
    # n_labels ={i:str(i)  for i in range(len(node))}

    G  = nx.Graph()
    G.add_nodes_from(node)

    edge = []
    #衛星為全連接的edge
    for i in range(node_s):
        for j in range(i+1, max_node):
            edge.append((node[i], node[j]))
    
    #UAV的edge
    if True:
        ground_group = [7,10,13,16,19,22]
        for i in range(1,7):
            edge.append((node[i], node[i+1]))
            edge.append((node[i], node[ground_group[i-1]]))
            edge.append((node[i], node[ground_group[i-1]+1 ]))
            edge.append((node[i], node[ground_group[i-1]+2 ]))
        # # uav1
        # # uav與uav
        # edge.append((node[1], node[2]))
        # # uav與地面
        # edge.append((node[1], node[7]))
        # edge.append((node[1], node[8]))
        # edge.append((node[1], node[9]))
        # # uav2
        # edge.append((node[2], node[3]))
        # edge.append((node[2], node[10]))
        # edge.append((node[2], node[11]))
        # edge.append((node[2], node[12]))
        # # uav3
        # edge.append((node[3], node[4]))
        # edge.append((node[3], node[13]))
        # edge.append((node[3], node[14]))
        # edge.append((node[3], node[15]))
        # # uav4
        # edge.append((node[4], node[5]))
        # edge.append((node[4], node[16]))
        # edge.append((node[4], node[17]))
        # edge.append((node[4], node[18]))
        # # uav5
        # edge.append((node[5], node[6])) 
        # edge.append((node[5], node[19])) 
        # edge.append((node[5], node[20])) 
        # edge.append((node[5], node[21])) 
        # # uav6
        # edge.append((node[6], node[1])) 
        # edge.append((node[6], node[22])) 
        # edge.append((node[6], node[23])) 
        # edge.append((node[6], node[24])) 
    
    #地面的edge
    if True:
        ground_group = [7,10,13,16,19,22]
        for n in ground_group:
            edge.append((node[n], node[n+1]))
            edge.append((node[n+1], node[n+2]))
            edge.append((node[n+2], node[n]))
        for n in ground_group:
            if n == ground_group[-1]:
                edge.append((node[n+2], node[ground_group[0]+2]))
            else:
                edge.append((node[n+2], node[n+2+3]))
        
    
    G.add_edges_from(edge)

    n_labels = {}
    e_labels = {}
    e_i_labels = {}
    pos = {}
    MEC_Node_Graph = []
    MEC_Link_Graph = []
    Link_propagation = []
    Link_arr = []
    share_link = []
    link_num = 0
    current_row = -1
    current_col = 0
    row_first_node = 0
    edge_type = [[],[],[],[],[]] # 衛星和HAP、衛星和地面、HAP和HAP、HAP和地面、地面和地面
    for i in range(max_node):
        n_labels[i] = str(i)
        key_lst = [k for k in G.adj[i]]
        MEC_Node_Graph.append(sorted(key_lst))
        if i < node_s:
            current_row = 5
            current_col = 5
        elif i < node_s + node_a:
            current_row = 4
            current_col = i+1 + (i-1)*5
        else:
            if i==3 or i==4:
                current_row = 3
                current_col = i-2 + (i-3)*7
            elif i==5 or i==6:
                current_row = 1
                current_col = i-4 + (i-5)*7
        pos[i] = (current_col, current_row)
                
        for j in key_lst:
            if e_labels.get((j, i), None) == None:
                if i < node_s:
                    if j<node_s:
                        print('只有一顆衛星')
                    elif j >= node_s and j < node_s+node_a: #衛星和HAP
                        edge_type[0].append(link_num)
                        e_labels[(i, j)] = round( s_a_dis *  ms_per_km, 3) # 490*0.0033 = 1.617
                    elif j >= node_s+node_a and j < max_node: #衛星和地面
                        edge_type[1].append(link_num)
                        e_labels[(i, j)] = round( s_g_dis *  ms_per_km, 3) # 500*0.0033 = 1.65
                elif i < node_s + node_a:
                    if j<node_s: #HAP和衛星 (上個if已經進去)
                        e_labels[(i, j)] = round( s_a_dis *  ms_per_km, 3) # 490*0.0033 = 1.617
                    elif j >= node_s and j < node_s+node_a: # HAP和HAP
                        edge_type[2].append(link_num)
                        e_labels[(i, j)] = round( a_a_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                    elif j >= node_s+node_a and j < max_node: # HAP和地面
                        edge_type[3].append(link_num)
                        e_labels[(i, j)] = round( a_g_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                else:
                    if j<node_s: #地面和衛星 (上上個if已經進去)
                        e_labels[(i, j)] = round( s_g_dis *  ms_per_km, 3) # 500*0.0033 = 1.65
                    elif j >= node_s and j < node_s+node_a: #地面和HAP (上上個if已經進去)
                        e_labels[(i, j)] = round( a_g_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                    elif j >= node_s+node_a and j < max_node: #地面和地面
                        edge_type[4].append(link_num)
                        if abs(i-j)>2: #內圈
                            e_labels[(i, j)] = round( g_g_dis *  0.005, 3) # 5*0.0033 = 0.0258
                        else: #同一個group, 同一個無人機管轄
                            e_labels[(i, j)] = round( g_g_dis_grougp *  0.005, 3) # 5*0.0033 = 0.0165
                       
                
                # print("e_labels[(i, j)] = ",e_labels[(i, j)])
                Link_propagation.append(e_labels[(i, j)])
                Link_arr.append([i,j])
                e_i_labels[(i, j)] = link_num # link index
                link_num += 1
        
        temp_lst = []
        for j in MEC_Node_Graph[i]:
            if e_i_labels.get((i, j), None) != None:
                temp_lst.append(e_i_labels[(i, j)])
            elif e_i_labels.get((j, i), None) != None:
                temp_lst.append(e_i_labels[(j, i)])
        MEC_Link_Graph.append(temp_lst)
        
    # print(pos)
    for i in range(node_s+node_a):
        share_link.append([])
        for j in range(len( MEC_Link_Graph[i])):
            check = True
            for k in range(len(share_link)):
                if MEC_Link_Graph[i][j] in share_link[k]:
                    check = False
                    break
            if check:
                share_link[i].append(MEC_Link_Graph[i][j])


    # plt.figure(figsize=(10, 10))
    # nx.draw(G, pos, with_labels=True, node_color='blue', node_size=1000, font_size=20, font_color="whitesmoke", font_weight="bold")
    # figure = plt.gcf()
    # figure.savefig(path + "SAGIN_graph")

    # plt.figure(figsize=(10, 10))
    # nx.draw_networkx_labels(G, pos, n_labels, font_size=12, font_color="whitesmoke", font_weight="bold")
    # nx.draw_networkx_edge_labels(G, pos, e_i_labels, font_size=12, verticalalignment = "bottom", rotate=False)
    # nx.draw(G, pos)
    # figure = plt.gcf()
    # figure.savefig(path + "SAGIN_graph_with_Link_idx")
    
    return MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num, share_link, Link_arr, edge_type

MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num, share_link, Link_arr, edge_type = generate_graph()
# print(MEC_Node_Graph)
# print(edge_type)
# print('====')
print(min(Link_propagation), max(Link_propagation))