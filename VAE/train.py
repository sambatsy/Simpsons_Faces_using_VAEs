device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz = batch[0].to(device)

net = VAE(10).to(device)

num_epochs = 10

lr = 1e-4

opt = optim.Adam(net.parameters(), lr=lr)

ls = []

for i in range(num_epochs):
    total_loss = 0

    for batch in train_loader:
        X = batch.to(device)
        Z, Xhat, mu, logvar = net(X)
        loss = get_loss(Xhat, X, mu, logvar)
        total_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    ls.append(total_loss)

plt.plot(ls)
