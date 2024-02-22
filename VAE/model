class VAE(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz = nz
        self.econv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.econv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.ebn2 = nn.BatchNorm2d(32)
        self.econv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.ebn3 = nn.BatchNorm2d(64)
        self.elinear1 = nn.Linear(64*8*8, 100)
        self.ebn4 = nn.BatchNorm1d(100)
        self.elinear2 = nn.Linear(100, self.nz)
        self.dlinear1 = nn.Linear(self.nz, 100)
        self.dbn1 = nn.BatchNorm1d(100)
        self.dlinear2 = nn.Linear(100, 64*8*8)
        self.dbn2 = nn.BatchNorm1d(64*8*8)
        self.dtconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dtconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1)
        self.dbn4 = nn.BatchNorm2d(16)
        self.dtconv3 = nn.ConvTranspose2d(16, 3, 3, 1, 1)
        self.plinear1 = nn.Linear(self.nz, self.nz)
        self.plinear2 = nn.Linear(self.nz, self.nz)

    def encoder_head(self, x):
        x = torch.relu(self.econv1(x))
        x = self.ebn1(x)
        x = torch.relu(self.econv2(x))
        x = self.ebn2(x)
        x = torch.relu(self.econv3(x))
        x = self.ebn3(x)
        x = x.view(-1, 64*8*8)
        x = torch.relu(self.elinear1(x))
        x = self.ebn4(x)
        x = torch.sigmoid(self.elinear2(x))
        return x

    def get_params(self, x):
        mu = self.plinear1(x)
        logvar = self.plinear2(x)
        return mu, logvar

    def encoder(self, x):
        x = self.encoder_head(x)
        mu, logvar = self.get_params(x)
        z = torch.exp(0.5 * logvar) * torch.randn_like(logvar) + mu
        return z, mu, logvar

    def decoder(self, x):
        x = torch.relu(self.dlinear1(x))
        x = self.dbn1(x)
        x = torch.relu(self.dlinear2(x))
        x = self.dbn2(x)
        x = x.view(-1, 64, 8, 8)
        x = torch.relu(self.dtconv1(x))
        x = self.dbn3(x)
        x = torch.relu(self.dtconv2(x))
        x = self.dbn4(x)
        x = torch.sigmoid(self.dtconv3(x))
        return x

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decoder(z)
        return z, x, mu, logvar
