# -*- coding="utf-8" -*-

from utils import  *
import parser
import random
from dataset import *
import capsgnn
import torch
import torch.optim as optim
import torch
import time
from layers import  margin_loss
from toolkit.logs.loggers import  Logger
import os
import sys
from torch.optim import lr_scheduler

def create_batches(train_graph_paths,args):
    return [train_graph_paths[i:i + args.batch_size]
            for i in range(0,len(train_graph_paths),args.batch_size)]


def main(args):

    train_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    sys.stdout = Logger(os.path.join("./log", "{}.log".format(train_time)))

    graph_train_path,graph_test_path,feature_map,target_map=load_data(args)

    #random.shuffle(graph_test_path)

    model=capsgnn.CapsGNN(args,len(feature_map),len(target_map))
    if torch.cuda.is_available() and args.gpu_id!=-1:
        model=model.cuda()

    optimizer=optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler=lr_scheduler.MultiStepLR(optimizer,[100,200],gamma=0.1)

    for epoch in range(args.epochs):
        random.shuffle(graph_train_path)
        batches=create_batches(graph_train_path,args)
        steps=len(batches)
        scheduler.step()
        for step in range(steps):
            start = time.time()
            accumulated_losses=0
            optimizer.zero_grad()
            batch=batches[step]
            for path in batch:
                data=create_input_data(path,feature_map,target_map)
                if torch.cuda.is_available() and args.gpu_id!=-1:
                    for key in data.keys():
                        data[key]=data[key].cuda()
                prediction,reconstruction_loss=model(data)
                loss= margin_loss(prediction, data["target"],args.lambd)+args.theta*reconstruction_loss
                accumulated_losses+=loss
            accumulated_losses = accumulated_losses / len(batch)
            accumulated_losses.backward()
            optimizer.step()
            print("epoch:[{}/{}],time:[{:.4f}],batch:[{}/{}],lr:[{}],loss:[{:.4f}]".
                  format(epoch,args.epochs,time.time()-start,step,steps,scheduler.get_lr()[0],
                         accumulated_losses))


        if (epoch+1)%50==0:
            test(model,graph_test_path,feature_map,target_map)




def test(model,test_graph_paths,feature_map,target_map):
    predictions=[]
    hits=[]
    random.shuffle(test_graph_paths)
    for path in tqdm(test_graph_paths):
        data=create_input_data(path,feature_map,target_map)
        for key in data.keys():
            data[key] = data[key].cuda()
        prediction, reconstruction_loss = model(data)
        prediction_mag = torch.sqrt((prediction ** 2).sum(dim=2))
        _, prediction_max_index = prediction_mag.max(dim=1)
        prediction = prediction_max_index.data.view(-1).item()
        predictions.append(prediction)
        hits.append(data["target"][prediction]==1.0)
    print("\nAccuracy"+str(round(np.mean(hits),4)))


if __name__=="__main__":
    args=parser.parameter_parser()
    torch.cuda.set_device(args.gpu_id)
    main(args)