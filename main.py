import os
import time
import argparse
import numpy as np
import torch
from utils import get_data_loader, save_checkpoint, load_checkpoint, mlp
from utils import parameters, evaluate_model, test_model, generate_label, collect_model
from loss import kd_loss, graphKL_loss, optimizing, graphKL_loss1
import warnings
from sklearn.metrics import f1_score, classification_report, confusion_matrix


warnings.filterwarnings("ignore")
torch.set_num_threads(1)


def train_student(args, models1,models2,data1, data2, device):

    _, _, feat1,adj1,label1, train_mask1, _, test_mask1,_, _ = data1
    _, _, feat2, adj2, label2, train_mask2, _, test_mask2, _, _ = data2

    loss_fcn = torch.nn.CrossEntropyLoss()

    t1_model = models1['t1_model']['model']
    t2_model = models1['t2_model']['model']
    t3_model = models1['t3_model']['model']
    t4_model = models1['t4_model']['model']
    s5_model = models2['s5_model']['model']
    s6_model = models2['s6_model']['model']

    s5_model_optimizer = torch.optim.Adam(s5_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    s6_model_optimizer = torch.optim.Adam(s6_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_recall=[]

    for epoch in range(args.s_epochs):


        s5_model.train()
        s6_model.train()

        logits1 = s5_model(feat1, adj1)
        logits2 = s6_model(feat2, adj2)


        logits_1t1 = generate_label(t1_model, feat1, adj1)
        logits_1t2 = generate_label(t2_model, feat1, adj1)
        logits_T1 = logits_1t1 + logits_1t2*0.5
        logits_2t1 = generate_label(t3_model, feat2, adj2)
        logits_2t2 = generate_label(t4_model, feat2, adj2)
        logits_T2 = logits_2t1 + logits_2t2*0.5

        class_loss1 = graphKL_loss1(logits1, logits_T1)
        class_loss2 = graphKL_loss1(logits1, logits_T2)
        class_loss3 = graphKL_loss1(logits2, logits_T1)
        class_loss4 = graphKL_loss1(logits2, logits_T2)
        class_loss = class_loss1 + class_loss2 + class_loss3 + class_loss4


        loss =class_loss

        s5_model_optimizer.zero_grad()
        s6_model_optimizer.zero_grad()


        with torch.autograd.enable_grad():
            loss.backward()

        s5_model_optimizer.step()
        s6_model_optimizer.step()

        print(f"Epoch {epoch:05d} | Loss: {loss:.4f} ")

        labels = label2
        logits = (logits1 + logits2) / 2

        _predictions = logits[test_mask2]
        _labels = labels[test_mask2]

        _predictions = np.argmax(_predictions.detach().cpu().numpy(), axis=1)
        print(f"F1-Score on testset:\n {confusion_matrix(_labels.detach().cpu().numpy(), _predictions)}")
        print(classification_report(_labels.detach().cpu().numpy(), _predictions, digits=4))
        report = classification_report(_labels.detach().cpu().numpy(), _predictions, output_dict=True)
        best_recalls = report['1']['recall']
        best_recall.append(best_recalls)

    print("Best Recall for Class 1:", max(best_recall))






def train_teacher(args, model, data, device):
    g, num_class, feat,adj,Label, train_mask, val_mask, test_mask,train_mask_anm, train_mask_norm = data
    loss_fcn = torch.nn.CrossEntropyLoss()#交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.t_epochs):
        model.train()
        loss_list = []
        feats = feat
        labels = Label
        logits = model(feats,adj)

        train_predictions = logits[train_mask]
        train_labels = labels[train_mask]

        train_predictions_anm = logits[train_mask_anm]
        train_labels_anm = labels[train_mask_anm]
        train_predictions_norm = logits[train_mask_norm]
        train_labels_norm = labels[train_mask_norm]

        loss1 = loss_fcn(train_predictions_anm, train_labels_anm)
        loss2=loss_fcn(train_predictions_norm, train_labels_norm)
        loss=loss2+loss1


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print(f"Epoch {epoch + 1:05d} | Loss: {loss_data:.4f}")

        if epoch % 10 == 0:
            train_predictions = np.argmax(train_predictions.detach().cpu().numpy(), axis=1)
            print(f"F1-Score on trainset:\n {confusion_matrix(train_labels.detach().cpu().numpy(), train_predictions)}")


def main(args):
    device = torch.device("cpu")
    data1, data2,data_info1,data_info2 = get_data_loader(args)
    model_dict1 = collect_model(args,data1, data_info1)
    model_dict2= collect_model(args,data2, data_info2)


    t1_model = model_dict1['t1_model']['model']
    t2_model = model_dict1['t2_model']['model']
    t3_model = model_dict1['t3_model']['model']
    t4_model = model_dict1['t4_model']['model']

    if os.path.isfile("./models/t1_model.pt"):
        load_checkpoint(t1_model, "./models/t1_model.pt", device)
    else:
        print("############ 1-train teacher1 #############")
        train_teacher(args, t1_model, data1, device)
        save_checkpoint(t1_model, "./models/t1_model.pt")

    if os.path.isfile("./models/t2_model.pt"):
        load_checkpoint(t2_model, "./models/t2_model.pt", device)
    else:
        print("############ 1-train teacher2 #############")
        train_teacher(args, t2_model, data1, device)
        save_checkpoint(t2_model, "./models/t2_model.pt")

    if os.path.isfile("./models/t3_model.pt"):
        load_checkpoint(t3_model, "./models/t3_model.pt", device)
    else:
        print("############ 2-train teacher1 #############")
        train_teacher(args, t3_model, data1, device)
        save_checkpoint(t3_model, "./models/t3_model.pt")

    if os.path.isfile("./models/t4_model.pt"):
        load_checkpoint(t4_model, "./models/t4_model.pt", device)
    else:
        print("############ 2-train teacher2 #############")
        train_teacher(args, t4_model, data1, device)
        save_checkpoint(t4_model, "./models/t4_model.pt")

    print("############ train student with teacher #############")
    train_student(args,model_dict1, model_dict2,data1, data2, device)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--t-epochs", type=int, default=160)
    parser.add_argument("--t-num-heads", type=int, default=4)
    parser.add_argument("--t-num-out-heads", type=int, default=6)
    parser.add_argument("--t1-num-layers", type=int, default=1)
    parser.add_argument("--t2-num-layers", type=int, default=2)
    parser.add_argument("--t3-num-layers", type=int, default=3)
    parser.add_argument("--t1-num-hidden", type=int, default=256)
    parser.add_argument("--t2-num-hidden", type=int, default=256)
    parser.add_argument("--t3-num-hidden", type=int, default=256)
    parser.add_argument("--t4-num-hidden", type=int, default=256)

    parser.add_argument("--s-epochs", type=int, default=560)
    parser.add_argument("--s-num-heads", type=int, default=2)
    parser.add_argument("--s-num-out-heads", type=int, default=2)
    parser.add_argument("--s-num-layers", type=int, default=4)
    parser.add_argument("--s-num-hidden", type=int, default=68)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--mode", type=str, default='mi')
    parser.add_argument("--train-mode", type=str, default='together')
    parser.add_argument('--loss-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=100, help="seed")
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    main(args)
    end = time.time()
    total_time = (end - start) / 60
    print("Total time: ", total_time, "min")
