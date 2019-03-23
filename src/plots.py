import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def plot_image(img, ax, title):
    ax.imshow(np.transpose(img, (1,2,0)) , interpolation='nearest')
    ax.set_title(title, fontsize=20)
    
def to_numpy(image, vsc):
    return torchvision.utils.make_grid(
        image.view(1, vsc.channels, vsc.height, vsc.width)
    ).cpu().detach().numpy()
    
def plot_encoding(image, vsc, latent_sz, alpha=None, width=1/7):
    image = vsc.transform(image).to(vsc.device)
    # decoded, mu, logvar, logspike = vsc.model.forward(image)
    decoded_params = vsc.model.forward(image)
    z = vsc.model.reparameterize(*decoded_params[1:])
    img = vsc.inverse_transform(vsc.model.decode(z))
    z = z.cpu().detach().numpy()[0]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(14,5))
    
    plot_image(to_numpy(image, vsc), ax0, 'Input Image')
    
    ax1.bar(np.arange(latent_sz), height=z, width=width, align='center')
    ax1.scatter(np.arange(latent_sz), z, color='blue')
    if alpha is not None:
        title = r"Latent Dimension %d - $\alpha$ = %.2f " % (latent_sz, alpha)
    else:
        title = r"Latent Dimension %d" % (latent_sz)
    ax1.set_title(title, fontsize=20)
    
    plot_image(to_numpy(img, vsc), ax2, 'Decoded Image')
    plt.subplots_adjust(hspace=0.5)

    
def plot_encoding_tcvae(image, vae, latent_sz, alpha=1, width=1/7):
    xs, x_params, zs, z_params = vae.reconstruct_img(image.to('cuda'))
    img = xs.cpu()[0]
    z = zs.cpu().detach().numpy()[0]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(14,5))
    
    plot_image(to_numpy(image, vae), ax0, 'Input Image')
    
    ax1.bar(np.arange(latent_sz), height=z, width=width, align='center')
    ax1.scatter(np.arange(latent_sz), z, color='blue')
    ax1.set_title(r"Latent Dimension %d - $\alpha$ = %.2f " % \
                  (latent_sz, alpha), fontsize=20)
    
    plot_image(to_numpy(img, vae), ax2, 'Decoded Image')
    plt.subplots_adjust(hspace=0.5)

    
    
def plot_horizontal_traversal(image, vsc, latent_sz, length, 
                              delta, threshold=1e-4, plot_all=False, 
                              plot_list=None, width=1/4, n_indices=15, plot=True):
    image = vsc.transform(image).to(vsc.device)
    # decoded, mu, logvar, logspike = vsc.model.forward(image)
    decoded_params = vsc.model.forward(image)
    z = vsc.model.reparameterize(*decoded_params[1:])
    img = vsc.inverse_transform(vsc.model.decode(z))
    z_ = z.cpu().detach().numpy()[0]
    
    if plot:
        plt.bar(np.arange(latent_sz), height=z_, width=width, align='center')
        plt.scatter(np.arange(latent_sz), z_, color='blue')
        plt.show()
    
    non_zero = [i for i in range(latent_sz) if np.abs(z_[i]) > threshold]
    inds = np.random.choice(non_zero, n_indices)
    if plot:
        print(inds)
    
    if not plot_all:
        non_zero = inds # [ind]
    if plot_list:
        non_zero = plot_list
    if plot:    
        print(non_zero)
    
    hor_traversal = []
    for ind in non_zero:
        images = []
        z1 = z.clone()
        for i in range(length):
            img = to_numpy(vsc.model.decode(z1), vsc)
            img = np.transpose(img, (1,2,0))
            img[:,0] = 1
            img[:,-1] = 1
            img[0,:] = 1
            img[-1,:] = 1
            images.append(img)
            z1[0, ind] = z1[0, ind] + delta if z[0,ind] < 0 else z1[0, ind] - delta

        hor_traversal.append(np.concatenate(images, axis=1))
    traversal = np.concatenate(hor_traversal, axis=0)
    if plot:
        plt.figure(figsize=(14,24))
        plt.axis('off')
        plt.imshow(traversal)
        plt.show()
    return traversal