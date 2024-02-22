import torchvision.utils as vutils
def plot_batch(batch: torch.Tensor,num : int, f_size,title:str):
    plt.figure(figsize=(f_size,f_size))
    plt.axis("off")
    plt.title(title)
    plot = plt.imshow(np.transpose(vutils.make_grid(batch.to(device)[:num], padding=0, normalize=True).cpu(), (1, 2, 0)), cmap='viridis')
    return plot

