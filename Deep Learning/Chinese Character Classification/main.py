import os
import logging
import warnings

from torch.serialization import save
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchsummary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib import dataset, readModel
import numpy as np
import datetime, time
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', UserWarning)

writer = SummaryWriter(log_dir='./log/tensorboard/')
dt_end = datetime.datetime.now()
time_start = dt_end.strftime('%Y%m%d_%H_%M_%S')
t_start = time.time()
best_acc = 0

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(logger, device, dataloader, model, model_name, loss_fn, optimizer, epochs, args, scheduler=None):
    model.train()
    loss_sum = 0
    correct  = 0
    total = 0

    for epoch in np.arange(1, epochs+1):
        loss_sum = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y= X.to(device), y.to(device)

            #compute loss
            pred = model(X)
            # Transfer GoogLeNetOutputs to Tensor
            if model_name == 'googlenet' and args.pretrained == False:
                pred = pred.logits
            loss = loss_fn(pred, y)
            l1_lambda = 1e-6
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += l1_reg * l1_lambda
            loss_sum += loss.item()

            #BackPropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += y.size(0)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss = loss_sum / (batch+1)
        train_acc = correct / total

        # check train info in tensorboard
        writer.add_scalar('Train_loss',train_loss, epoch)
        writer.add_scalar('Train_acc', train_acc, epoch)

        logger.info(f"Train_loss: {train_loss:>7f}  Train_acc: {train_acc:>7f}  [{epoch}/{epochs}]")
        if scheduler:
            scheduler.step()

def evaluate_cross_validation(logger, device, train_dataset,model):

    train_dataloader = DataLoader(train_dataset, shuffle=True)
    x, y = train_dataloader()
    cv = KFold(len(y), K,shuffle=True,random_state=0)
    scores = cross_val_score(model,x,y,cv=cv)
    print(scores)
    print ("Mean score: {} (+/-{})".format( np.mean (scores),sem(scores)))
    print(np.mean(scores), "+-", np.std(simple_cv_scores))

def test(logger, device, dataloader, model, epochs):
    global best_acc
    size = 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X,y  = X.to(device), y.to(device)
            pred = model(X)
            size += len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        accuracy = correct/size
        logger.info(f"Test_acc: {accuracy:>7f}")

    # Save checkpoint.
    acc = 100.*accuracy/size
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epochs,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def main(args):
    seed_flg=True
    seed = 3
    seed_set(seed,seed_flg)
    if seed_flg:
        seed_everything(seed)

    try:
        os.remove(args.logdir)
    except:
        pass
    #logger
    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(filename=args.logdir))

    #datasets & dataloaders
    train_dataset = dataset.EikllxDataset(args.root, 'Train.csv', size = args.img_size)
    test_dataset = dataset.EikllxDataset(args.root, 'Test.csv', size = args.img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # print current time to logfile
    logger.info('Start:' + time_start + '\n')
    logger.info('Seed Set: '+str(seed_flg)+ ' , Seed :'+ str(seed))
    logger.info(args)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\nUsing {device} device"+'\n')

    #define model
    model = readModel.get_model(model_name=args.model, use_pretrained = args.pretrained).to(device)
    # logger.info(torchsummary.summary(model, (3, args.img_size, args.img_size)))
    logger.info(model)

    if device == 'cuda':
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    #loss function & optimizer
    loss_criterion = torch.nn.CrossEntropyLoss()
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


    #learning rate scheduler
    class CosineDecayScheduler:
        def __init__(self, max_epochs: int, warmup_lr_limit=10, warmup_epochs=int(args.n_epochs/10)):
            self._max_epochs = max_epochs
            self._warmup_lr_limit = warmup_lr_limit
            self._warmup_epochs = warmup_epochs

        def __call__(self, epoch: int):
            epoch = max(epoch, 1)
            if epoch <= self._warmup_epochs:
                return self._warmup_lr_limit * epoch / self._warmup_epochs
            epoch -= 1
            rad = np.pi * epoch / self._max_epochs
            weight = (np.cos(rad) + 1.) / 2
            return self._warmup_lr_limit * weight
    lr_scheduler_func = CosineDecayScheduler(args.n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                    lr_lambda=lr_scheduler_func)

    #train model
    train(logger, device, train_dataloader, model, args.model, loss_criterion,
                    optimizer, args.n_epochs, args, scheduler=lr_scheduler)

    # evaluate_cross_validation(logger, device, train_dataset,model)

    #test model
    test(logger, device, test_dataloader, model, args.n_epochs)

    #end time
    dt_end = datetime.datetime.now()
    t_end = time.time()
    time_end = dt_end.strftime('%Y%m%d_%H_%M_%S')
    time_sum = t_end - t_start
    imple_time = str(int((time_sum / 60))) + 'm ' + str(int(time_sum % 60)) + 's'
    logger.info('\nEnd:' + time_end)
    logger.info('Implementation time:' + imple_time + '\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CCT-4 Training')
    parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
    parser.add_argument('--root', default='./data', type=str,
                        help='/path/to/dataset')
    parser.add_argument('--logdir', default='./log/'+time_start+'.log', type=str,
                        help='/path/to/log')
    # Model
    parser.add_argument('--model', default='resnet18_ft', type=str)

    # Optimization
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', '-bs', default=64, type=int)
    parser.add_argument('--img_size', default=256, type=int)

    args = parser.parse_args()

    main(args)
