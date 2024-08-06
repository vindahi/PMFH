from torch.utils.data import DataLoader
from utils import *

def train_val_test_split(args, dataset, train_idxs, idcs_retrieval, idcs_query):
    train_L, train_x, train_y = dataset[0], dataset[1], dataset[2]
    retrieval_L, retrieval_x, retrieval_y = dataset[3], dataset[4], dataset[5]
    query_L, query_x, query_y = dataset[6], dataset[7], dataset[8]

    train_X_list, train_Y_list, train_L_list = [], [], []
    retrieval_X_list, retrieval_Y_list, retrieval_L_list = [], [], []
    query_X_list, query_Y_list, query_L_list = [], [], []

    for idx in train_idxs:
        train_X_list.append(train_x[idx])
        train_Y_list.append(train_y[idx])
        train_L_list.append(train_L[idx])
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    train_L = np.array(train_L_list)

    for idx in idcs_retrieval:
        retrieval_X_list.append(retrieval_x[idx])
        retrieval_Y_list.append(retrieval_y[idx])
        retrieval_L_list.append(retrieval_L[idx])
    
    retrieval_x_for_this_client = np.array(retrieval_X_list)
    retrieval_y_for_this_client = np.array(retrieval_Y_list)
    retrieval_L_for_this_client = np.array(retrieval_L_list)

    for idx in idcs_query:
        query_X_list.append(query_x[idx])
        query_Y_list.append(query_y[idx])
        query_L_list.append(query_L[idx])
    
    query_x_for_this_client = np.array(query_X_list)
    query_y_for_this_client = np.array(query_Y_list)
    query_L_for_this_client = np.array(query_L_list)

    imgs = {'train': train_X, 'query': query_x_for_this_client, 'database': retrieval_x_for_this_client}
    texts = {'train': train_Y, 'query': query_y_for_this_client, 'database': retrieval_y_for_this_client}
    labels = {'train': train_L, 'query': query_L_for_this_client, 'database': retrieval_L_for_this_client}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train','query','database']}
    shuffle = {'train': True, 'query': False, 'database': False}
    dataloader = {x: DataLoader(dataset[x], batch_size = args.batch_size, shuffle=shuffle[x], num_workers=4) for x in ['train','query','database']}
    trainloader = dataloader['train']
    testloader = dataloader['query']
    databaseloader = dataloader['database']
    return trainloader, testloader, databaseloader









