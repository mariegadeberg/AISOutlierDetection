from model import VRNN
import torch
from torch.utils.data import DataLoader
from utils_preprocess import AISDataset
from Config import *
from utils_preprocess import TruncCollate
from utils_train import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
import csv
import matplotlib.pyplot as plt
# Setting arguments

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default="./data/", help="Which directory to find the 'data' folder containing training, validation and test")
parser.add_argument("--num_epoch", type=int, default=1, help="How many epochs should run during training")
parser.add_argument("--beta", type=int, default=1, help="The size of the regularization coefficient 'beta'")
parser.add_argument("--save_dir", type=str, default="./models/", help="Directory to save model")
parser.add_argument("--print_every", type=int, default=1, help="Determines how often it print to terminal. Default every 10th epoch")
parser.add_argument("--train", type=str, default="train.pcl", help="What training data should be used")
parser.add_argument("--val", type=str, default="val.pcl", help="What training data should be used")
parser.add_argument("--ROI", type=str, default="blt", help="Specify the region of interest")
parser.add_argument("--batchsize", type=int, default=1)

args = parser.parse_args()

num_epoch = args.num_epoch
beta = args.beta
path = args.path
save_dir = args.save_dir
print_every = args.print_every
train_ = args.train
val_ = args.val
ROI = args.ROI
batchsize = args.batchsize

input_shape = Config.input_shape[ROI]
latent_shape = Config.latent_shape
lr = Config.lr

# Setup for training
writer = SummaryWriter()

model = VRNN(input_shape, latent_shape, beta)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epoch = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

train_ds = AISDataset(path+train_, path+"mean_"+ROI+".pcl")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, collate_fn=TruncCollate())

val_ds = AISDataset(path+val_, path+"mean_"+ROI+".pcl")
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, shuffle=True, collate_fn=TruncCollate())

# move the model to the device
model = model.to(device)
model.apply(init_weights)

diagnostics_list = []

with open(save_dir+f"output_{num_epoch}{ROI}.txt", "w") as output_file:
    header = ["training_loss", "validation_loss", "training_kl", "validation_kl", "training_logpx", "validation_logpx"]
    csv_writer = csv.DictWriter(output_file, fieldnames=header)
    csv_writer.writeheader()

    print(f"Training initialized...")
    while epoch < num_epoch:
        print(f"--> Training started for epoch {epoch}")
        epoch_train_loss = 0
        epoch_val_loss = 0

        epoch_train_kl = 0
        epoch_val_kl = 0

        epoch_train_logpx = 0
        epoch_val_logpx = 0

        #w_ave = {"phi_x.0.weight":[],
        #         "phi_z.0.weight":[],
        #         "prior.0.weight":[],
        #         "encoder.0.weight":[],
        #         "decoder.0.weight":[],
        #         "rnn.weight_ih_l0":[],
        #         "rnn.weight_hh_l0":[]}

        #loss_plot = []

        model.train()
        i = 0
        for inputs in train_loader:

            inputs = inputs.to(device)

            if i % 1000 == 0:
                print(f"     passing input {i}/{len(train_loader)}")

            loss, diagnostics = model(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #plot_grad_flow(model.named_parameters())
            #w_ave = get_weights(w_ave, model)

            if i % 1000 == 0:
                diagnostics_list.append(diagnostics)

            epoch_train_kl += np.mean(diagnostics["kl"])
            epoch_train_logpx += np.mean(diagnostics["log_px"])

            epoch_train_loss += loss.item()
            #loss_plot.append(loss.item())

            i += 1

        print(f"--> Validation started for epoch {epoch}")

        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("KL/train", epoch_train_kl, epoch)
        writer.add_scalar("Log_px/train", epoch_train_logpx, epoch)

        model.eval()

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)

                loss, diagnostics = model(inputs)

                epoch_val_kl += np.mean(diagnostics["kl"])
                epoch_val_logpx += np.mean(diagnostics["log_px"])

                epoch_val_loss += loss.item()

            print("Validation done")
            writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
            writer.add_scalar("KL/validation", epoch_val_kl, epoch)
            writer.add_scalar("Log_px/validation", epoch_val_logpx, epoch)

        #Prepare output
        training_loss = epoch_train_loss/len(train_loader)
        val_loss = epoch_val_loss/len(val_loader)

        training_kl = epoch_train_kl/len(train_loader)
        val_kl = epoch_val_kl/len(val_loader)

        training_logpx = epoch_train_logpx/len(train_loader)
        val_logpx = epoch_val_logpx/len(train_loader)

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, training loss: {training_loss:.4f}, validation loss: {val_loss:.4f}')

        csv_writer.writerow({"training_loss": training_loss,
                             "validation_loss": val_loss,
                             "training_kl": training_kl,
                             "validation_kl": val_kl,
                             "training_logpx": training_logpx,
                             "validation_logpx": val_logpx})

        output_file.flush()
        writer.flush()

        epoch += 1

#plt.tight_layout()
#plt.savefig(save_dir+"/gradient_bars.png")
##
#legend = []
#plt.figure()
#for name in w_ave.keys():
#    plt.plot(w_ave[name])
#    legend.append([name])
#plt.title("Average of gradients through small dataset")
#plt.legend(legend)
#plt.savefig(save_dir+"/gradient_flow.png")
##
#plt.figure()
#plt.plot(loss_plot)
#plt.title("Training loss through small training set")
#plt.show()


writer.close()

torch.save(model.state_dict(), save_dir+f"vrnn_{ROI}{num_epoch}_epochs.pt")

with open(save_dir+f"diagnostics_{num_epoch}_{ROI}.pcl", "wb") as fp:
    pickle.dump(diagnostics_list, fp)



