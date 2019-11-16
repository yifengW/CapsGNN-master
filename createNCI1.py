# -*- coding="utf-8" -*-

import os
import sys
import json
import glob
import shutil
import random
from utils import *


def create_nodenumber_mapping(labels,edges):
    mapping={value:str(i) for i,value in  enumerate(labels.keys())}
    mapping_labels=dict()
    mapping_edges=list()
    for value in labels.keys():
        mapping_labels[mapping[value]]=str(labels[value])

    for edge in edges:
        edge[0]=int(mapping[str(edge[0])])
        edge[1]=int(mapping[str(edge[1])])
        mapping_edges.append([edge[0],edge[1]])

    return mapping_labels,mapping_edges


def create_KCI1_json(adj_path,indicator_path,graph_label_path,node_label_path,output_path):
    """

    :param adj_path:
    :param indicator_path:
    :param graph_label_path:
    :param node_label_path:
    :param output_path:
    :return:
    """
    adjfile=open(adj_path,"r",encoding="utf-8")
    indicatorfile=open(indicator_path,"r",encoding="utf-8")
    graphlabelfile=open(graph_label_path,"r",encoding="utf-8")
    nodelabelfile=open(node_label_path,"r",encoding="utf-8")
    adj=adjfile.readlines()
    indicator=indicatorfile.readlines()
    graphlabel=graphlabelfile.readlines()
    nodellabel=nodelabelfile.readlines()

    indicatordict=create_indicator_mapping(indicator)
    graphdict=create_graph_mapping(graphlabel)
    nodedict=create_node_mapping(nodellabel)
    adjdict=create_adj_mapping(adj)

    for indicator  in indicatordict.keys():
        target=graphdict[int(indicator)]
        labels=dict()
        output_json=dict()
        for nodenumber  in  indicatordict[indicator]:
            labels[str(nodenumber)]=nodedict[nodenumber]

        edges=list()
        for node1 in  indicatordict[indicator]:
            if node1 in adjdict.keys():
                for  node2  in adjdict[node1]:
                     temp=sorted([node1,node2],key=lambda x:x)
                     if temp not in edges:
                        edges.append(temp)

        labels,edges=create_nodenumber_mapping(labels,edges)
        output_json["edges"]=edges
        output_json["target"]=target
        output_json["labels"]=labels

        output_json_path=output_path+indicator+".json"
        print(output_json_path)
        outputfile=open(output_json_path,"w",encoding="utf-8")
        json.dump(output_json,outputfile)
        outputfile.close()

def split_train_test(output_path):
    root="/media/data3/wyf/code/graph-master/dataKCI1"
    all_json=glob.glob(output_path+"*.json")
    train_json=random.sample(all_json,int(len(all_json)*0.9))
    test_json=list()
    for j in all_json:
        if j not in train_json:
            test_json.append(j)

    #train
    for i  in train_json:
        save_name=root+"/train/"+i.split('/')[-1]
        print(save_name)
        shutil.copy(i,save_name)

    for i in test_json:
        save_name = root + "/test/" + i.split('/')[-1]
        print(save_name)
        shutil.copy(i,save_name)


if __name__=="__main__":
    adj_path="./NCI1/NCI1_A.txt"
    indicator_path="./NCI1/NCI1_graph_indicator.txt"
    graph_label_path="./NCI1/NCI1_graph_labels.txt"
    node_label_path="./NCI1/NCI1_node_labels.txt"
    output_path="./outputNCI1/"
    #create_KCI1_json(adj_path,indicator_path,graph_label_path,node_label_path,output_path)
    split_train_test(output_path)
    pass