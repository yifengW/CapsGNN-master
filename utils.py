# -*- coding="utf-8" -*-

def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value:i for i, value in enumerate(node_properties)}


def create_indicator_mapping(indicator):
    indicatordict={value.strip('\n'):[] for _,value in enumerate(indicator)}
    for i,value in enumerate(indicator):
        value=value.strip('\n')
        indicatordict[value].append(i+1)
    return indicatordict


def create_graph_mapping(graphlabel):
    return {i+1:int(value.strip('\n')) for i,value in enumerate(graphlabel)}


def create_node_mapping(nodelabel):
    return {i+1:int(value.strip('\n')) for i,value in enumerate(nodelabel)}



def create_adj_mapping(adj):
    adjdict={int(value.strip('\n').split(',')[0]):[] for i,value in enumerate(adj)}
    for i,value in enumerate(adj):
        value=value.strip('\n')
        adjdict[int(value.split(',')[0])].append(int(value.split(',')[1]))

    return adjdict