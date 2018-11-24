from model import SimpleLSTM
from data_loader import VAT_video_Dataset
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.stats import pearsonr
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
def ccc(y_true, y_pred):
    true_mean = torch.mean(y_true)
    #true_variance = torch.var(y_true)
    pred_mean = torch.mean(y_pred)
    #pred_variance = torch.var(y_pred)
    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = torch.std(y_pred)
    std_gt = torch.std(y_true)
    rho = torch.FloatTensor(np.array(rho))
    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)
    return ccc 

def batch_ccc (y_trues, y_preds, lens):
    # y: (n_batch, n_seq,)
    y_true_pack = []
    y_pred_pack = []
    for y_true, y_pred, length  in zip(y_trues, y_preds, lens):
        y_true = y_true[:length]
        y_pred = y_pred[:length]
        y_true_pack.extend(y_true)
        y_pred_pack.extend(y_pred)
    y_true_pack = torch.from_numpy(np.asarray(y_true_pack))
    y_pred_pack = torch.from_numpy(np.asarray(y_pred_pack))
    
    return ccc(y_true_pack, y_pred_pack)
        
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    batch_losses = []
    batch_ccces = []
    for batch_idx, (data, target, lens) in enumerate(train_loader):
        data, target, lens = data.to(device), target.to(device), lens.to(device)
        data, target, lens = data.float(), target.float(), lens.to(device) # amake sure input is float, not double
        optimizer.zero_grad()
        #output = model(data).type(torch.cuda.DoubleTensor)
        output = model(data, lens)
        loss = model.loss(target, output.squeeze(),  lens)
        with torch.no_grad():
            train_ccc = batch_ccc(target, output.squeeze(), lens)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCCC: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler), #the train set length is not len(train_loader.dataset)
                100. * batch_idx / len(train_loader), loss.item(), train_ccc))
        batch_losses.append(loss.item())
        batch_ccces.append(train_ccc)
    return sum(batch_losses)/(batch_idx+1), sum(batch_ccces)/(batch_idx+1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_ccc = 0
    assert len(test_loader)==1 # assume test once'val_indices.pkl'
    with torch.no_grad():
        for data, target, lens in test_loader:
            data, target, lens = data.to(device), target.to(device), lens.to(device)
            data, target, lens = data.float(), target.float(), lens.to(device) # amake sure input is float, not double
            output = model(data, lens)
            loss = model.loss(target,  output.squeeze(), lens)
        
            test_loss += loss
            test_ccc += batch_ccc(target, output.squeeze(), lens)
    test_loss /= len(test_loader.sampler)
#    test_ccc /= len(test_loader.sampler)
    print('\nTest set: Average loss: {:.4f}, Average CCC: {:.4f} '.format(test_loss, test_ccc))
    return test_loss, test_ccc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='decision tree Example')
    parser.add_argument('--label_name', type=str, default='arousal', 
                        help='label name')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='adam momentum (default: 0.8)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                           help='how many batches to wait before logging training status')
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")    
    args = parser.parse_args()
    global args
    
    if args.label_name =='arousal':
        pretrained_xgb_model = 'xgb_ccc:0.2234_label:arousal.pkl'
    else:
        pretrained_xgb_model = 'xgb_ccc:0.3599_label:valence.pkl'
    visual_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/vgg_fer_features_fps=15_fc7'
    audio_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/egemaps_VAD'
    text_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/MPQA' 
    train_dataset = VAT_video_Dataset(visual_root_path, audio_root_path, text_root_path, '../train_dict.pkl',  
                                label_name=args.label_name, feature_selection_from_model= pretrained_xgb_model)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    
    feature,_,_ = train_dataset[0]
    n_feature = feature.shape[1]
    val_dataset = VAT_video_Dataset(visual_root_path, audio_root_path, text_root_path, '../val_dict.pkl',
                                    label_name=args.label_name, feature_selection_from_model= pretrained_xgb_model)
    
    validation_loader = DataLoader(dataset=val_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    model = SimpleLSTM(n_feature=n_feature, n_hidden=256, n_out=1, nb_layers=1, on_gpu=use_cuda).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer, milestones=[50,100,150, 200 ], gamma=0.2)
    
    history = {}
    history['loss'] = []
    history['private_val_loss'] = []
    history['ccc'] = []
    history['private_val_ccc'] = []
    for epoch in np.arange(args.epochs)+1:
        scheduler.step()
        train_loss, train_ccc = train( model, device, train_loader, optimizer, epoch)
        val_loss, val_ccc = test(model, device,validation_loader )
        history['loss'].append(train_loss)
        history['private_val_loss'].append(val_loss)
        history['ccc'].append(train_ccc)
        history['private_val_ccc'].append(val_ccc)
    
    # test on test set
    print("Evaluation on official validation set.")
    test_loss, test_ccc = test( model, device, validation_loader)
    torch.save(model, args.label_name+'{}_loss{:.4f}_ccc{:.4f}.pb'.format(args.epochs, test_loss, test_ccc))
    # summarize history for loss
    plt.plot(history['loss'], color='b', label='Training')
    plt.plot(history['private_val_loss'], color='g', label='Private Validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='lower left')
    plt.savefig('train_val_loss'+args.label+'_{}_loss{:.4f}_ccc{:.4f}.pb'.format(args.epochs, test_loss, test_ccc)+'.png')
    plt.show()
    
    # summarize history for accuracy
    plt.plot(history['ccc'] , color='b', label='Training')
    plt.plot(history['private_val_ccc'], color='g', label='Private Validation')
    plt.title('CCC metric')
    plt.ylabel('CCC')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig('train_val_ccc'+args.label+'_{}_loss{:.4f}_ccc{:.4f}.pb'.format(args.epochs, test_loss, test_ccc)+'.png')
    plt.show()