import numpy as np
import urllib.request
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datafile = 'bloodmnist_64'

# Download the dataset to the local folder
if not os.path.isfile(f"./{datafile}.npz"):
    urllib.request.urlretrieve(f"https://zenodo.org/records/10519652/files/{datafile}.npz?download=1", f"{datafile}.npz" )

# Decompress and load
dataset = np.load(f"./{datafile}.npz")

def plot_classes(images, labels, image_shape, N_rows = 2):
    class_ids, class_first_occur = np.unique(labels, return_index=True)
    N_cols = int(np.ceil(len(class_ids) / N_rows))
    fig, ax = plt.subplots(N_rows, N_cols, sharex=True, sharey=True)
    ax = ax.reshape((N_rows, N_cols))
    for i in range(N_rows):
        for j in range(N_cols):
            if i * N_cols + j < len(class_ids):
                idx = class_first_occur[i * N_cols + j]
                label = labels[idx]
                ax[i, j].set_title(f"Class {label}")
                ax[i, j].set_yticks([])
                ax[i, j].set_xticks([])
                ax[i, j].imshow(images[idx], cmap="gray")
            else:
                ax[i, j].axis("off")
    plt.show()

image_shape = dataset['train_images'][0].shape
# Show the classes of blood in the dataset
plot_classes(dataset['train_images'], dataset['train_labels'], image_shape, N_rows=2)

# Extract testing, training, and validation sets
trainImages = dataset['train_images']
testImages = dataset['test_images']
valImages = dataset['val_images']


# Define our auto-encoder architecture (technically a VAE)
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelEncoder = nn.Sequential(
            # Several conv layers to learn image features (with stride
            # to reduce dimensionality)
            nn.Conv2d(3, 8, 5),
            # ReLU since this is a deep architecture (so we want to reduce vanishing gradient)
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride = 2),
            nn.ReLU(),
            nn.Flatten(),
            # And linear feed-forward layers to use them
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, 32)
        self.fc_logvar = nn.Linear(128, 32)
        self.modelDecoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, 5, stride = 2, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 5, stride = 2, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride = 2, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute([0, 3, 2, 1])
            hidden = self.modelEncoder(x)
            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + std * eps
            return self.modelDecoder(latent).permute([0, 3, 2, 1]), mu, logvar

    def encode(self, x):
        hidden = self.modelEncoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # Latent encoding is a normal distribution
        return torch.cat((mu, logvar), axis=1)


def vae_loss(reconstructed_x, x, mu, logvar, beta = 1.0):
    mse_recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    # KL divergence between the latent distribution and standard normal distribution (i.e. mean 0 and std. dev 1)
    kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
    )
    # Balance reconstruction and latent space regularization
    return mse_recon_loss + kl_loss * beta

model = AutoEncoder()
if os.path.isfile("./weights.bin"):
    model.load_state_dict(torch.load("./weights.bin"))
model = model.to(device)

optimiser = torch.optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(trainImages.astype('float32'), batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valImages.astype('float32'), batch_size=256, shuffle=True)
# lots of epochs as this is a fairly deep architecture, with Conv layers
N_Epochs = 200
for i in range(N_Epochs):
    model.train(True)
    for x in train_loader:
        x = x.to(device) / 255
        predicted, mu, logvar = model(x)
        loss = vae_loss(predicted, x, mu, logvar)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    avg_loss = 0.0
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device) / 255

            pred, mu, logvar = model(xb)

            avg_loss += vae_loss(pred, xb, mu, logvar).item()

        avg_loss /= len(val_loader)
    print(f"Epoch: {i + 1}\nValidation Loss: {avg_loss}")

model.train(False)
torch.save(model.state_dict(), "./weights.bin")
reconstructedImages = model(torch.from_numpy(testImages.astype('float32') / 255).to(device))[0].cpu().detach().numpy()
autoEncodedTrainImages = model.encode(torch.from_numpy(trainImages.astype('float32') / 255).to(device).permute([0, 3, 2, 1])).cpu().detach().numpy()
autoEncodedTestImages = model.encode(torch.from_numpy(testImages.astype('float32') / 255).to(device).permute([0, 3, 2, 1])).cpu().detach().numpy()
actualLabels = dataset['test_labels']

# Cluster with k-means clustering on the auto-encoded sets
kMeansAuto = KMeans(n_clusters = 8)
kMeansAuto.fit(autoEncodedTrainImages)
kMeansAutoPredictions = kMeansAuto.predict(autoEncodedTestImages)

# Cluster on the non-encoded sets
kMeansImage = KMeans(n_clusters = 8)
kMeansImage.fit(trainImages.reshape((trainImages.shape[0], trainImages.shape[1] * trainImages.shape[2] * 3)))
imagePredictions = kMeansImage.predict(testImages.reshape(testImages.shape[0], testImages.shape[1] * testImages.shape[2] * 3))

# Show the K-means clustering with autoencoded
f, axarr = plt.subplots(4,8)
f.suptitle("Auto-Encoder with k-means clustering")

for i in range(8):
  axarr[0][i].set_title(f"Class {i}")

# Show 4 images per class (i.e. to see how well the clustering works)
for i in range(4):
  for j in range(8):
    axarr[i][j].imshow(testImages[np.argwhere(kMeansAutoPredictions == j)[i][0]])
plt.show()


# Then show with no encoding
f, axarr = plt.subplots(4,8)
f.suptitle("No encoding with k-means clustering")

for i in range(8):
  axarr[0][i].set_title(f"Class {i}")

for i in range(4):
  for j in range(8):
    axarr[i][j].imshow(testImages[np.argwhere(imagePredictions == j)[i][0]])
plt.show()


tsne = TSNE(2)
autoEncoder2d = tsne.fit_transform(autoEncodedTestImages)
f, axarr = plt.subplots(1,2, figsize=(25, 20))
for i in range(8):
    autoIndices = np.argwhere(kMeansAutoPredictions == i)
    trueIndices = np.argwhere(actualLabels == i)
    axarr[0].scatter(autoEncoder2d[autoIndices, 0], autoEncoder2d[autoIndices, 1], label=f"Autoencoder Label: {i}")
    axarr[1].scatter(autoEncoder2d[trueIndices, 0], autoEncoder2d[trueIndices, 1], label=f"Autoencoder True Label: {i}")
axarr[0].legend()
axarr[1].legend()
plt.show()
