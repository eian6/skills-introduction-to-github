import os
from sklearn.metrics import f1_score, confusion_matrix
from dgl import DGLGraph
import numpy as np
import torch
import random
import dgl
from gnns import GCN1,GCN2,GCN3,GCN4,GCN5,GCN6
from topo_semantic import get_loc_model

def parameters(model):
    num_params = 0
    for params in model.parameters():
        cur = 1
        for size in params.data.shape:
            cur *= size
        num_params += cur
    return num_params

def evaluate(feats, model, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
    model.train()

    return score, loss_data.item()


def test_model(data,mask,logits1,logits2):

    g, _, feat,  adj, labels,_, _, _, _, _= data
    labels =labels
    logits=(logits1+logits2)/2

    _predictions = logits[mask]
    _labels = labels[mask]

    _predictions = np.argmax(_predictions.detach().cpu().numpy(), axis=1)
    print(f"F1-Score on testset:\n {confusion_matrix(_labels.detach().cpu().numpy(), _predictions)}")

    return None


def generate_label(t_model, feats,adj):
    t_model.eval()
    logits_t = t_model(feats,adj)
    return logits_t


def evaluate_model(valid_dataloader, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, feats, labels = valid_data
            feats = feats.to(device)
            labels = labels.to(device)
            score, val_loss = evaluate(feats.float(), s_model, subgraph, labels.float(), loss_fcn)
            score_list.append(score)
            val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def get_teacher(args, data_info):
    heads1 = ([args.t_num_heads] * args.t1_num_layers) + [args.t_num_out_heads]
    heads2 = ([args.t_num_heads] * args.t2_num_layers) + [args.t_num_out_heads]
    heads3 = ([args.t_num_heads] * args.t3_num_layers) + [args.t_num_out_heads]
    model1 = GCN1(data_info['num_feats'], args.t1_num_hidden)
    model2 = GCN2(data_info['num_feats'], args.t2_num_hidden)
    model3 = GCN3(data_info['num_feats'], args.t3_num_hidden)
    model4 = GCN4(data_info['num_feats'], args.t4_num_hidden)
    return model1, model2, model3,model4


def get_student(args, data_info):
    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    model5 = GCN5(data_info['num_feats'], args.s_num_hidden)
    model6 = GCN6(data_info['num_feats'], args.s_num_hidden)
    return model5,model6


def mlp(dim, logits, device):
    output = logits
    return output


def get_feat_info(args):
    feat_info = {}
    feat_info['s_feat'] = [args.s_num_heads * args.s_num_hidden] * args.s_num_layers
    feat_info['t1_feat'] = [args.t_num_heads * args.t1_num_hidden] * args.t1_num_layers
    feat_info['t2_feat'] = [args.t_num_heads * args.t2_num_hidden] * args.t2_num_layers
    feat_info['t3_feat'] = [args.t_num_heads * args.t3_num_hidden] * args.t3_num_layers
    return feat_info

def edge_drop(graph, drop_rate):

    edges = list(graph.edges())
    num_edges_to_drop = int(len(edges) * drop_rate)
    edges_to_drop = random.sample(edges, num_edges_to_drop)

    new_graph = graph.copy()
    for edge in edges_to_drop:
        new_graph.remove_edge(*edge)

    return new_graph
def get_data_loader(args):

    feat = np.loadtxt('./dataset/automotive/svd_u.txt', delimiter=' ')
    feat = torch.Tensor(feat)
    adj = np.loadtxt('./dataset/automotive/weights.txt', delimiter=' ')
    adj = torch.Tensor(adj)

    label = np.load('dataset/automotive/labels.npy').astype('int32')
    num_class = np.unique(label).size
    Label = label.flatten()
    Label = torch.LongTensor(Label)


    adj_tensor = torch.Tensor(adj)
    edge_index = adj_tensor.nonzero().t()
    edge_attr = adj_tensor[edge_index[0], edge_index[1]]


    g = DGLGraph()
    g.add_nodes(feat.shape[0])
    g.ndata['x'] = torch.Tensor(feat)
    g.add_edges(edge_index[0], edge_index[1])
    g.edata['edge_attr'] = torch.Tensor(edge_attr)

    g.ndata['label'] = torch.Tensor(Label.float())

    g2 = DGLGraph()
    g2.add_nodes(feat.shape[0])
    g2.ndata['x'] = torch.Tensor(feat)
    g2.add_edges(edge_index[0], edge_index[1])
    g2.edata['edge_attr'] = torch.Tensor(edge_attr)

    g2.ndata['label'] = torch.Tensor(Label.float())

    train_size =0.6
    val_size =0
    train_mask = torch.zeros(g.ndata['label'].shape).bool()
    val_mask = torch.zeros(g.ndata['label'].shape).bool()
    test_mask = torch.zeros(g.ndata['label'].shape).bool()
    train_mask_anm = torch.zeros(g.ndata['label'].shape).bool()
    train_mask_norm = torch.zeros(g.ndata['label'].shape).bool()
    anm_list = (g.ndata['label']).nonzero(as_tuple=True)[0]
    norm_list = (g.ndata['label'] == 0).nonzero(as_tuple=True)[0]

    anm_id_list = torch.Tensor.tolist(anm_list)
    norm_id_list = torch.Tensor.tolist(norm_list)

    num_anm = len(anm_id_list)
    num_norm = len(norm_id_list)


    random.seed(42)
    train_anm_id = random.sample(anm_id_list, int(num_anm * train_size))
    train_norm_id = random.sample(norm_id_list, int(num_norm * train_size))
    anm_id_list = list(set(anm_id_list) - set(train_anm_id))
    norm_id_list = list(set(norm_id_list) - set(train_norm_id))
    val_anm_id = random.sample(anm_id_list, int(num_anm * val_size))
    val_norm_id = random.sample(norm_id_list, int(num_norm * val_size))
    test_anm_id = list(set(anm_id_list) - set(val_anm_id))
    test_norm_id = list(set(norm_id_list) - set(val_norm_id))

    train_mask[train_anm_id] = True
    train_mask[train_norm_id] = True
    val_mask[val_anm_id] = True
    val_mask[val_norm_id] = True
    test_mask[test_anm_id] = True
    test_mask[test_norm_id] = True
    train_mask_anm[train_anm_id] = True
    train_mask_norm[train_norm_id] = True



    n_classes = num_class
    num_feats = feat.shape[1]
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats

    mean = 0
    stddev = 0.01
    noise = np.random.normal(mean, stddev, adj.shape)
    noise = torch.Tensor(noise)
    adj = adj + noise

    drop_rate = 0.2  # 可根据需要调整边丢弃概率
    g2 = edge_drop(g2, drop_rate)
    new_adj = torch.zeros_like(adj)
    src, dst = g.edges()
    edge_attrs = g.edata['edge_attr']
    new_adj[src, dst] = edge_attrs

    return (g, num_class, feat,adj,Label, train_mask, val_mask, test_mask,train_mask_anm, train_mask_norm),\
        (g2, num_class, feat,new_adj,Label, train_mask, val_mask, test_mask,train_mask_anm, train_mask_norm),\
        data_info,data_info

def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")


def collect_model(args,data, data_info):
    device = torch.device("cpu")
    g, num_class, feat, adj, Label, train_mask, val_mask, test_mask,_,_=data
    feat_info = get_feat_info(args)

    t1_model, t2_model, t3_model,t4_model = get_teacher(args, data_info)
    t1_model.to(device)
    t2_model.to(device)
    t3_model.to(device)
    t4_model.to(device)


    s5_model,s6_model = get_student(args, data_info)
    s5_model.to(device)
    s6_model.to(device)


    local_model = get_loc_model(feat_info,adj)
    local_model.to(device)
    local_model_s = get_loc_model(feat_info, adj)
    local_model_s.to(device)


    s5_model_optimizer = torch.optim.Adam(s5_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s6_model_optimizer = torch.optim.Adam(s6_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t1_model_optimizer = torch.optim.Adam(t1_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t2_model_optimizer = torch.optim.Adam(t2_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t3_model_optimizer = torch.optim.Adam(t3_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t4_model_optimizer = torch.optim.Adam(t4_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    local_model_optimizer = None
    local_model_s_optimizer = None


    model_dict = {}
    model_dict['s5_model'] = {'model': s5_model, 'optimizer': s5_model_optimizer}
    model_dict['s6_model'] = {'model': s6_model, 'optimizer': s6_model_optimizer}
    model_dict['local_model'] = {'model': local_model, 'optimizer': local_model_optimizer}
    model_dict['local_model_s'] = {'model': local_model_s, 'optimizer': local_model_s_optimizer}
    model_dict['t1_model'] = {'model': t1_model, 'optimizer': t1_model_optimizer}
    model_dict['t2_model'] = {'model': t2_model, 'optimizer': t2_model_optimizer}
    model_dict['t3_model'] = {'model': t3_model, 'optimizer': t3_model_optimizer}
    model_dict['t4_model'] = {'model': t4_model, 'optimizer': t4_model_optimizer}

    return model_dict