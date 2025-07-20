from losses.Orthogonal_regularization_loss import orthonormal_loss
from losses.SCL import SpectralContrastiveLoss
from model.PSD_estimation import PSD_estimation
from model.BeatFormer import BeatFormer
from torch import optim
from tensorboardX import SummaryWriter
import datetime, argparse
from dataloading.Dataloader import *

torch.manual_seed(42)
np.random.seed(42)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    date_time = datetime.datetime.now()
    date = str(date_time).split(' ')[0]
    hour = str(date_time).split(' ')[1].split('.')[0].split(':')
    folder_name = date + '_' + hour[0] + 'h_' + hour[1] + 'min_' + args.model_name
    training_folder = args.dest_folder + folder_name + '/'

    rppg_model = BeatFormer(args.channels, args.temporal_length, args.window_size, args.image_size, args.image_size,
                            args.overlap, args.heads, args.blocks, args.dropout, args.fs).to(device)
    rppg_model = rppg_model.to(device)
    SCL_loss = SpectralContrastiveLoss(args.margin, args.scaling)

    Train = rPPG_Dataloader(data_path=args.data_path, dataset=args.train_dataset, temporal_length=args.temporal_length,
                            temporal_stride=args.temporal_stride, face_mode=args.face_mode, img_size=args.image_size,
                            split_set='Train')

    train_dataloader = torch.utils.data.DataLoader(dataset=Train, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.num_workers, drop_last=True)

    optimizer_rppg = optim.AdamW(list(rppg_model.parameters()), lr=args.lr_model, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_rppg, len(train_dataloader.dataset), eta_min=args.min_eta)

    if not os.path.exists(os.path.dirname(training_folder)):
        os.makedirs(os.path.dirname(training_folder))

    writer = SummaryWriter(training_folder + 'logs_' + args.model_name + '/')
    save_model = training_folder + 'saved_models/'
    print('Initial learning rate: {:.8f}'.format(optimizer_rppg.state_dict()['param_groups'][0]['lr']))

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        running_loss = 0
        for x, (train_images, _, _, train_name) in enumerate(train_dataloader, 0):
            rppg_model.train()
            train_images = train_images.to(device)
            optimizer_rppg.zero_grad()
            with torch.set_grad_enabled(True):

                output_rgb, weights_rgb = rppg_model(train_images[:, 0, :, :, :, :])
                output_rgb = (output_rgb - torch.mean(output_rgb)) / torch.std(output_rgb)
                psd_rgb = PSD_estimation(output_rgb, args.fs)

                output_hsv, weights_hsv = rppg_model(train_images[:, 1, :, :, :, :])
                output_hsv = (output_hsv - torch.mean(output_hsv)) / torch.std(output_hsv)
                psd_hsv = PSD_estimation(output_hsv, args.fs)

                output_lab, weights_lab = rppg_model(train_images[:, 2, :, :, :])
                output_lab = (output_lab - torch.mean(output_lab)) / torch.std(output_lab)
                psd_lab = PSD_estimation(output_lab, args.fs)

                output_flip, weights_flip = rppg_model(train_images[:, 3, :, :, :, :])
                output_flip = (output_flip - torch.mean(output_flip)) / torch.std(output_flip)
                psd_flip = PSD_estimation(output_flip, args.fs)

                output_occ, weights_occ = rppg_model(train_images[:, 4, :, :, :, :])
                output_occ = (output_occ - torch.mean(output_occ)) / torch.std(output_occ)
                psd_occ = PSD_estimation(output_occ, args.fs)

                psds = torch.stack([psd_rgb, psd_hsv, psd_lab, psd_flip, psd_occ]).permute(1, 0, 2)
                weights = torch.stack([weights_rgb, weights_hsv, weights_lab, weights_flip, weights_occ])
                loss_ortho = orthonormal_loss(weights) * args.alpha
                loss_c = SCL_loss(psds)
                loss = loss_c + loss_ortho
                print("{}/{} loss : {:.3f}, loss_c : {:.3f}, loss_ortho : {:.3f},".format(x, int(len(
                    train_dataloader.dataset) / args.batch_size), loss.item(), (loss_c.item()), loss_ortho.item()))
                loss.backward()
                optimizer_rppg.step()
            running_loss += loss.item()
        epoch_loss = running_loss / int(len(train_dataloader.dataset)/args.batch_size)
        print('Loss : {:.4f}'.format(epoch_loss))
        scheduler.step()
        writer.add_scalars(args.train_dataset + '_Loss', {'Train loss': epoch_loss}, epoch)
        date_time = datetime.datetime.now()
        date = str(date_time).split(' ')[0]
        if not os.path.exists(os.path.dirname(save_model)):
            os.makedirs(os.path.dirname(save_model))
        torch.save(rppg_model.state_dict(), os.path.join(save_model,
                                                         date + '_train=' + args.train_dataset + '_arch=' + args.model_name + '_epoch=%d' % epoch + '_epoch_loss=%f' % epoch_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training UNET Derivative model")
    parser.add_argument('--channels', type=int, help='number of channels')
    parser.add_argument('--window_size', type=int, help='temporal window size')
    parser.add_argument('--overlap', type=int, help='patch embedding overlap')
    parser.add_argument('--heads', type=int, help='number of heads')
    parser.add_argument('--blocks', type=int, help='number of ZOCA blocks')
    parser.add_argument('--dropout', type=int, help='dropout model')
    parser.add_argument('--weight_decay', type=int, help='weight decay')
    parser.add_argument('--min_eta', type=int, help='min LR')
    parser.add_argument('--image_size', type=int, help='input image resize')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--face_mode', type=int, help='face_mode')
    parser.add_argument('--scaling', type=int, help='scaling factor')
    parser.add_argument('--margin', type=int, help='margin parameter')
    parser.add_argument('--alpha', type=float, help='orthonormal regularization factor')
    parser.add_argument('--num_workers', type=int, help='num_workers dataloader')
    parser.add_argument('--train_dataset', type=str, help='dataset for training')
    parser.add_argument('--temporal_length', type=int, help="temporal input sequence legnth")
    parser.add_argument('--temporal_stride', type=int, help="temporal input sequence stride")
    parser.add_argument('--lr_model', type=float, help='learning rate model')
    parser.add_argument('--model_name', type=str, help='Name training folder to save')
    parser.add_argument('--data_path', type=str, help='dataset directory')
    parser.add_argument('--dest_folder', type=str, help='directory to  save training')
    parser.add_argument('--fs', type=int, help='Sampling rate dataset')
    parser.add_argument('--epochs', type=int, help='total training epochs')
    args = parser.parse_args()
    train()
