# -*- coding="utf-8" -*-

import glob
import json
import torch
import random
import numpy as np
import pandas as pd
#from utils import create_numeric_mapping
from layers import ListModule, PrimaryCapsuleLayer, Attention, SecondaryCapsuleLayer, margin_loss
from pygcn import  GraphConvolution
from pygcn import GCN

class CapsGNN(torch.nn.Module):
    def __init__(self,args,number_of_features,number_of_targets):
        super(CapsGNN,self).__init__()

        self.args=args
        self.number_of_features=number_of_features
        self.number_of_targets=number_of_targets
        self._setup_layers()

    def _setup_base_layers(self):
        self.base_layers = [GraphConvolution(self.number_of_features, self.args.gcn_filters)]
        for layer in range(self.args.gcn_layers - 1):
            self.base_layers.append(GraphConvolution(self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        self.first_capsule = PrimaryCapsuleLayer(in_units=self.args.gcn_filters, in_channels=self.args.gcn_layers,
                                                 num_units=self.args.gcn_layers,
                                                 capsule_dimensions=self.args.capsule_dimensions)

    def _setup_attention(self):
        self.attention=Attention(self.args.gcn_layers*self.args.capsule_dimensions,
                                 self.args.inner_attention_dimension)

    def _setup_class_capsule(self):
        self.class_capsule=SecondaryCapsuleLayer(self.args.capsule_dimensions,
                                                 self.args.number_of_capsules,
                                                 self.number_of_targets, self.args.capsule_dimensions)

    def _setup_reconstruction_layers(self):
        self.reconstruction_layer_1 = torch.nn.Linear(self.number_of_targets * self.args.capsule_dimensions,
                                                      int((self.number_of_features * 2) / 3))
        self.reconstruction_layer_2 = torch.nn.Linear(int((self.number_of_features * 2) / 3),
                                                      int((self.number_of_features * 3) / 2))
        self.reconstruction_layer_3 = torch.nn.Linear(int((self.number_of_features * 3) / 2), self.number_of_features)

    def _setup_graph_capsules(self):
        self.graph_capsule = SecondaryCapsuleLayer(self.args.gcn_layers, self.args.capsule_dimensions,
                                                   self.args.number_of_capsules, self.args.capsule_dimensions)

    def _setup_layers(self):
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()

    def calculate_reconstruction_loss(self, capsule_input, features):
        """
        Calculating the reconstruction loss of the model.
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """

        v_mag = torch.sqrt((capsule_input ** 2).sum(dim=1))
        _, v_max_index = v_mag.max(dim=0)
        v_max_index = v_max_index.data

        capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
        if torch.cuda.is_available() and self.args.gpu_id!=-1:
            capsule_masked =capsule_masked.cuda()

        capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
        capsule_masked = capsule_masked.view(1, -1)

        #feature_counts = features.sum(dim=0)
        #feature_counts = feature_counts / feature_counts.sum()

        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(capsule_masked))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.softmax(self.reconstruction_layer_3(reconstruction_output), dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)

        reconstruction_loss = torch.sum((features - reconstruction_output) ** 2)

        return reconstruction_loss



    def forward(self,data):
        features=data["features"]
        edges=data["edges"]
        hidden_representations=[]
        for layer in self.base_layers:
            features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters, -1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1, self.args.gcn_layers * self.args.capsule_dimensions)
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers,
                                                                     self.args.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)

        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions,
                                                                  self.args.number_of_capsules)
        class_capsule_output = self.class_capsule(reshaped_graph_capsule_output)
        class_capsule_output = class_capsule_output.view(-1, self.number_of_targets * self.args.capsule_dimensions)
        class_capsule_output = torch.mean(class_capsule_output, dim=0).view(1, self.number_of_targets,
                                                                            self.args.capsule_dimensions)
        reconstruction_loss = self.calculate_reconstruction_loss(
            class_capsule_output.view(self.number_of_targets, self.args.capsule_dimensions), data["features"])
        return class_capsule_output, reconstruction_loss


if __name__=="__main__":
    import parser
    arg=parser.parameter_parser()
    torch.cuda.set_device(arg.gpu_id)
    model=CapsGNN(arg,8,10).cuda()
    data=dict()
    data["features"]=torch.rand(20,8).cuda()
    data["edges"]=torch.rand(20,20).cuda()
    output=model(data)
    pass