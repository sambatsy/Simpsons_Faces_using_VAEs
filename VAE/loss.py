def get_loss(x, xhat, mu, logvar):
    m = 1e-4
    recon_loss = F.mse_loss(x,xhat)
    kld_loss = torch.mean(torch.sum(-0.5*(1+logvar - torch.exp(logvar)- mu**2), dim=1), dim=0)
    loss = m*kld_loss + (1-m)*recon_loss
    return loss
