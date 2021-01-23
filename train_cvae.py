from cvae import CVAE
from utils_train import *
from utils_preprocess import *
from Config import *
import argparse
import csv
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default="./data/", help="Which directory to find the 'data' folder containing training, validation and test")
parser.add_argument("--num_epoch", type=int, default=1, help="How many epochs should run during training")
parser.add_argument("--save_dir", type=str, default="./models/", help="Directory to save model")
parser.add_argument("--train", type=str, default="train.pcl", help="What training data should be used")
parser.add_argument("--val", type=str, default="val.pcl", help="What training data should be used")
parser.add_argument("--ROI", type=str, default="bh", help="Specify the region of interest")
parser.add_argument("--batchsize", type=int, default=32, help="Batchsize to use")
parser.add_argument("--latent_features", type=int, default=100, help="Number of latent features in VAE")
parser.add_argument("--normalize_input", type=str, default="True", choices={"True", "False"}, help="Whether to use a normalized input or merely binary values if the ship was present or not")
parser.add_argument("--name", type=str, default="", help="name to individualize output files")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

if args.normalize_input == "True":
    train_ds = AISDataset_Image(args.path + args.train, Config)
elif args.normalize_input == "False":
    train_ds = AISDataset_ImageOneHot(args.path + args.train, Config)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)

if args.normalize_input == "True":
    val_ds = AISDataset_Image(args.path + args.val, Config)
elif args.normalize_input == "False":
    val_ds = AISDataset_ImageOneHot(args.path + args.val, Config)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batchsize, shuffle=True)

model = CVAE(Config.latent_shape)

optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
# move the model to the device
model = model.to(device)

training_data = defaultdict(list)
validation_data = defaultdict(list)
epoch=0

grads = []

with open(args.save_dir+f"output_{args.num_epoch}{args.ROI}{args.name}.txt", "w") as output_file:
    header = ["training_elbo", "validation_elbo", "training_kl", "validation_kl", "training_logpx", "validation_logpx", "mi", "au"]
    csv_writer = csv.DictWriter(output_file, fieldnames=header)
    csv_writer.writeheader()

    while epoch < args.num_epoch:
        print(f"Runninng epoch {epoch + 1}")

        w_ave = {"encoder.0.weight": [],
                 "encoder.2.weight": [],
                 "encoder.3.weight": [],
                 "encoder.5.weight": [],
                 "encoder.6.weight": [],
                 "encoder.8.weight": [],
                 "encoder.9.weight": [],
                 "encoder.11.weight": [],
                 "encoder.12.weight": [],
                 "encoder.15.weight": [],
                 "encoder.17.weight": [],
                 "decoder.0.weight": [],
                 "decoder.2.weight": [],
                 "decoder.3.weight": [],
                 "decoder.4.weight": [],
                 "decoder.5.weight": [],
                 "decoder.7.weight": [],
                 "decoder.8.weight": [],
                 "decoder.10.weight": [],
                 "decoder.11.weight": [],
                 "decoder.13.weight": [],
                 "decoder.15.weight": []
                 }

        training_epoch_data = defaultdict(list)
        validation_epoch_data = defaultdict(list)
        model.train()

        for inputs in train_loader:
            inputs = inputs.unsqueeze(1) #adding greyscale channel
            inputs = inputs.to(device)

            loss, diagnostics = model(inputs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            #plot_grad_flow(model.named_parameters())
            w_ave = get_weights(w_ave, model)

            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]


        ##for k, v in training_epoch_data.items():
        #    training_data[k] += [np.mean(training_epoch_data[k])]

        model.eval()
        with torch.no_grad():

            num_examples = 0
            mi = 0
            for val_inputs in val_loader:

                val_inputs = val_inputs.unsqueeze(1)  # adding greyscale channel
                val_inputs = val_inputs.to(device)

                loss, diagnostics = model(val_inputs)

                for k, v in diagnostics.items():
                    validation_epoch_data[k] += [v.mean().item()]

                num_examples += val_inputs.size(0)
                mutual_info = model.calc_mi(val_inputs)
                mi += mutual_info * val_inputs.size(0)

            mi = mi / num_examples
            au, _ = calc_au(model, val_loader, device)

            #for k, v in validation_epoch_data.items():
            #    validation_data[k] += [np.mean(validation_epoch_data[k])]

        print(f"{'training elbo':6} | mean = {np.mean(training_epoch_data['elbo']):10.3f}")
        print(f"{'training KL':6} | mean = {np.mean(training_epoch_data['kl']):10.3f}")
        print(f"{'mi':6} | {mi}")
        print(f"{'au':6} | {au}")

        csv_writer.writerow({"training_elbo": np.mean(training_epoch_data["elbo"]),
                             "validation_elbo": np.mean(validation_epoch_data["elbo"]),
                             "training_kl": np.mean(training_epoch_data["kl"]),
                             "validation_kl": np.mean(validation_epoch_data["kl"]),
                             "training_logpx": np.mean(training_epoch_data["log_px"]),
                             "validation_logpx": np.mean(validation_epoch_data["log_px"]),
                             "mi": mi,
                             "au": au
                             })

        output_file.flush()

        grads.append(w_ave.copy())

        epoch += 1

#plt.tight_layout()
#plt.show()
#
#legend = []
#plt.figure()
#for name in w_ave.keys():
#    plt.plot(w_ave[name])
#    legend.append([name])
#plt.title("Trace of gradients through 1st epoch of training")
#plt.legend(legend)
#plt.xlabel("Steps")
#plt.ylabel("Parameter value")
##plt.ylim(-0.05, 0.05)
##plt.savefig(save_dir+"/gradient_flow_no_mean_zoom.eps")
#plt.show()


torch.save(model.state_dict(), args.save_dir+f"cvae_{args.ROI}{args.num_epoch}_epochs{args.name}.pt")

with open(args.save_dir+f"grad_{args.num_epoch}_{args.ROI}{args.name}.pcl", "wb") as fp:
    pickle.dump(grads, fp)


#with open(args.save_dir+f"training_{args.num_epoch}_{args.ROI}.pcl", "wb") as fp:
#    pickle.dump(training_data, fp)

#with open(args.save_dir+f"validation_{args.num_epoch}_{args.ROI}.pcl", "wb") as fp:
#    pickle.dump(validation_data, fp)

