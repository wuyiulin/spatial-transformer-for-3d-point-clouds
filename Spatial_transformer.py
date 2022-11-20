#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import pdb
import torch.nn as nn


class offset_deform(nn.Module):
    "offset_deform"    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(67, 3, kernel_size=1, bias=False),
                        nn.BatchNorm2d(3),
                        nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(134, 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(negative_slope=0.2))
        self.convedge_64 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2))
        self.convedge_128 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2))
        self.convedge_1024 = nn.Sequential(nn.Conv2d(128, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=0.2))
                                                #input_edge_feature
        self.conv2d_out1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2))
                                                #input_out1
        self.conv2d_out2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2))
                                                #cat_max_mean_1
        self.conv2d_out3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2))
        self.weights = torch.randn((256,3*3), dtype=torch.float32, requires_grad=True)
        self.biases = torch.randn((3*3), dtype=torch.float32, requires_grad=True)
        self.fc1 = nn.Linear(1024, 512) # net need change
        self.fc2 = nn.Linear(512, 256)

    def pairwise_distance(self, point_cloud):
        """
        Args:
        
        point_cloud: tensor (batch_size, num_points, num_dims)
        
        Returns:
        
        pairwise distance: (batch_size, num_points, num_points)
        """
        
        og_batch_size = point_cloud.shape[0]
        point_cloud = torch.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = torch.unsqueeze(point_cloud, 0)
        
        
        point_cloud_transpose = point_cloud.permute( 0, 2, 1)
        point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2*point_cloud_inner
        point_cloud_square = torch.sum(torch.square(point_cloud), dim=-1, keepdim=True)
        point_cloud_square_tranpose = point_cloud_square.permute( 0, 2, 1)
        #pdb.set_trace()
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    
    
    def knn(self, adj_matrix, k=20):
        """
            Get KNN based on the pairwise distance.
        Args:
            pairwise distance: (batch_size, num_points, num_points)
            k: int
        Returns:
            nearest neighbors: (batch_size, num_points, k)
        """
        neg_adj = -adj_matrix
        _, nn_idx = torch.topk(neg_adj, k=k)
        return nn_idx


    def get_edge_feature(self, point_cloud, nn_idx, k=20):
        """
        Construct edge feature for each point
        Args:
            point_cloud: (batch_size, num_points, 1, num_dims)
            nn_idx: (batch_size, num_points, k)
            k: int
        Returns:
            edge features: (batch_size, num_points, k, num_dims)
        """
        og_batch_size = point_cloud.shape[0]
        point_cloud = torch.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = torch.unsqueeze(point_cloud, 0)

        point_cloud_central = point_cloud

        point_cloud_shape = point_cloud.shape
        batch_size = point_cloud_shape[0]
        num_points = point_cloud_shape[1]
        num_dims = point_cloud_shape[2]
        
        idx_ = torch.arange(0,batch_size) * num_points
        idx_ = torch.reshape(idx_, (batch_size, 1, 1))

        idx_ = idx_.cuda().int()
        
        point_cloud_flat = torch.reshape(point_cloud, (-1, num_dims))
        #point_cloud_flat = point_cloud
        ak = nn_idx+idx_
        
        #point_cloud_neighbors = torch.gather(point_cloud_flat ,0 ,nn_idx+idx_)
        point_cloud_neighbors = point_cloud_flat[ak, :]
        point_cloud_central = torch.unsqueeze(point_cloud_central, -2)

        point_cloud_central = torch.tile(point_cloud_central, (1, 1, k, 1))
        edge_feature = torch.cat((point_cloud_central, (point_cloud_neighbors-point_cloud_central)), dim=-1)
        return edge_feature

    def input_transform_net(self, edge_feature, is_training, bn_decay=None, K=3, is_dist=False):
        """ Input (XYZ) Transform Net, input is BxNx3 gray image
                Return:
            Transformation matrix of size 3xK """
        batch_size = edge_feature.shape[0]
        num_point = edge_feature.shape[1]
        maxpooling2d = nn.MaxPool2d(kernel_size=[num_point, 1])
        
        fc1 = nn.Linear(batch_size, 512) # net need change
        fc2 = nn.Linear(512, 256)
        # input_image = tf.expand_dims(point_cloud, -1)
        ###############################################
        edge_feature = edge_feature.permute(0, 3, 1, 2)
        ###############################################
        
        net = self.convedge_64(edge_feature)
        net = self.convedge_128(net)
        net,__ = torch.max(net, dim=-1, keepdim=True)
        net = self.convedge_1024(net)
        #pdb.set_trace()
        ##############################
        #net = net.permute(0, 2, 3, 1)
        ##############################
        
        net = maxpooling2d(net)
        
        
        net = torch.reshape(net, (batch_size, -1))
        
        net = self.fc1(net)
        net = self.fc2(net)
        
        #self.biases += torch.Tensor(np.eye(K).flatten(), dtype=torch.float32)
        biases = (torch.from_numpy(np.eye(K).flatten()) + self.biases).cuda()
        weights = self.weights.cuda()
        transform = torch.matmul(net, weights)
        transform = transform + biases
        transform = torch.reshape(transform, (batch_size, K, K))
        transform = transform.float()
        return transform

        

    def pre_inputs_feature(self, point_cloud):

        point_cloud = point_cloud.permute( 0, 2, 1)
        batch_size = point_cloud.shape[0]
        num_point = point_cloud.shape[1]
        input_image = torch.unsqueeze(point_cloud, -1)

        k = 30

        adj = self.pairwise_distance(point_cloud)
        nn_idx = self.knn(adj, k=k)
        edge_feature = self.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
        
        transform = self.input_transform_net(edge_feature, is_training=True, bn_decay=None, K=3, is_dist=True)
        point_cloud_transformed = torch.matmul(point_cloud, transform)
        input_image = torch.unsqueeze(point_cloud_transformed, -2)
        adj = self.pairwise_distance(point_cloud_transformed)
        nn_idx = self.knn(adj, k=k)

        edge_feature = self.get_edge_feature(input_image, nn_idx=nn_idx, k=k)
        edge_feature = edge_feature.permute( 0, 3, 1, 2)
        
        out1 = self.conv2d_out1(edge_feature)
        
        out2 = self.conv2d_out2(out1)
        out2 = out2.permute( 0, 2, 3, 1)
        net_max_1,__ = torch.max(out2, dim=-2, keepdim=True)
        net_mean_1 = torch.mean(out2, dim=-2, keepdim=True)
        cat_max_mean_1 = torch.cat((net_max_1, net_mean_1), dim=-1)
        cat_max_mean_1 = cat_max_mean_1.permute( 0, 3, 1, 2)
        
        out3 = self.conv2d_out3(cat_max_mean_1)
        out3 = out3.permute( 0, 2, 3, 1)
        
        return input_image, out3

    def forward(self, input_image, feature=None, num_graph=4, num_feat=16, num_neighbor = 30, K=3, is_training=True, weight_decay=0.0, bn_decay=None, visualize=False):
        out5_all = []
        net_max_all = []
        net_mean_all = []
        trans_all = []

        inputs, feature = self.pre_inputs_feature(point_cloud = input_image)

        for i in range(num_graph):
            IF = torch.cat((inputs, feature), dim=-1)
            IFP = IF.permute( 0, 3, 1, 2)
            #trans1 = self.conv1(torch.cat((inputs, feature), dim=-1))
            trans1 = self.conv1(IFP)
            trans1 = trans1.permute( 0, 2, 3, 1)
            adj = self.pairwise_distance(torch.squeeze(trans1, dim=-2))
            nn_idx = self.knn(adj, k=num_neighbor)
            
            edge_feature = self.get_edge_feature(IF, nn_idx=nn_idx, k=num_neighbor)
            
            edge_feature = edge_feature.permute( 0, 3, 1, 2)
            
            out4 = self.conv2(edge_feature)
            
            out4 = out4.permute( 0, 2, 3, 1)
            
            net_max_2,__ = torch.max(out4, dim = -2, keepdim = True)
            net_mean_2 = torch.mean(out4, dim = -2, keepdim = True)
            
            o5 = torch.cat((net_max_2, net_mean_2), dim=-1)
            o5 = o5.permute( 0, 3, 1, 2)
            
            #out5 = self.conv3(torch.cat((net_max_2, net_mean_2), dim=-1))
            out5 = self.conv3(o5)
            out5 = out5.permute( 0, 2, 3, 1)

            out5_all.append(out5)
            net_max_all.append(net_max_2)
            net_mean_all.append(net_mean_2)
            trans_all.append(trans1)
            

        out5_all = torch.cat(out5_all, -1)
        net_max_all = torch.cat(net_max_all, -1)
        net_mean_all = torch.cat(net_mean_all, -1)
        out5_all = out5_all.permute( 0, 3, 1, 2)
        net_max_all = net_max_all.permute( 0, 3, 1, 2)
        net_mean_all = net_mean_all.permute( 0, 3, 1, 2)
        if not visualize:
            return out5_all, net_max_all, net_mean_all
        else:
            return out5_all, net_max_all, net_mean_all, trans_all

    

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
