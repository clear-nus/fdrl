import torch
from torch.nn.functional import logsigmoid
import torch.nn.utils.spectral_norm as sn


def flow(x, model, steps, eta, noise_factor, f_divergence):

    x_k = torch.autograd.Variable(x.clone(), requires_grad=True)

    for _ in range(steps):
        
        nn_out = model(x_k)

        # for LSIF-Pearson, nn_out == R
        if f_divergence == 'Pearson':
            f_prime = 2*nn_out
        
        # for LR-KL, LR-JS, LR-logD, nn_out == logR
        elif f_divergence == 'KL':
            f_prime = nn_out
        elif f_divergence == 'JS':
            f_prime = logsigmoid(nn_out) # log R/(1+R)
        elif f_divergence == 'logD':
            f_prime = -logsigmoid(-nn_out) # log(R+1)
        else:
            print("f_divergence is not valid")
        
        grad = torch.autograd.grad(f_prime.sum(), [x_k], retain_graph=True)[0]
        x_k.data -= eta * grad + noise_factor * torch.randn_like(x_k)
    
    return x_k.detach()


def get_loss(loss_function, model_x_nu, model_x_de):

    if loss_function == 'LSIF':
        # model predicts R(x)
        r_x_nu = model_x_nu
        r_x_de = model_x_de
        first = 0.5 * (r_x_de ** 2)
        second = -r_x_nu

    elif loss_function == 'KL':
        # model predicts log R(x)
        r_x_nu = torch.exp(model_x_nu)
        r_x_de = torch.exp(model_x_de)
        first = r_x_de
        second = -model_x_nu

    elif loss_function == 'LR':
        # model predicts log R(x)
        r_x_nu = torch.exp(model_x_nu)
        r_x_de = torch.exp(model_x_de)
        first = -logsigmoid(-model_x_de) # log 1/1+r
        second = -model_x_nu - logsigmoid(-model_x_nu) # log r/1+r, equivalent to -logsigmoid(model_x_nu)

    else:
        print("DR loss function not specified!")

    loss = (first + second).mean()
    r_x_nu = r_x_nu.mean().item()
    r_x_de = r_x_de.mean().item()
    # r_x_nu = r_x_nu if not np.isnan(r_x_nu) else 0.
    # r_x_de = r_x_de if not np.isnan(r_x_de) else 0.
    first = first.mean().item()
    second = second.mean().item()
    return loss, r_x_nu, r_x_de, first, second


def SpectralNorm(module, apply=True):
    if apply:
        return sn(module)
    else:
        return module