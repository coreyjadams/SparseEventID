import torch
import pytorch_lightning as pl

# torch.set_float32_matmul_precision('high')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src.config.optimizer import LossBalanceScheme

from src import logging

logger = logging.getLogger("NEXT")

# import torchmin
import scipy
import numpy

# These are the functions that describe the form of the data in energy:
def gauss(x, amp, mu, sigma):
    return amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

def exp(x, a0, tau):
    return a0 * numpy.exp(-x*tau) 

def expgauss(x, a0, tau, amp, mu, sigma):
    return exp(x, a0, tau) + gauss(x, amp, mu, sigma)
    # return a0 * numpy.exp(x*tau) + amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

def torchmin_expgauss(x, params):

    # Exponential, fit vals are the amplitude, shift of x, and decay const
    e_val =  params[0] * numpy.exp(-(x )*params[1]) 
    # e_val =  params[0] * numpy.exp(-(x - params[2])*params[1]) 
    # return e_val
    # print("Exp component: ", e_val)
    # Gaussian, fit is amp, mean, and sigma
    g_val = params[3] * numpy.exp(-0.5*(x-params[4])**2./params[5]**2.)
    # print("Gauss component: ", g_val)
    return e_val + g_val


def get_errors(cov):
    """
    Find errors from covariance matrix
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of the fit parameters.
    Returns
    -------
    err : 1-dim np.ndarray
        Errors asociated to the fit parameters.
    """
    return numpy.sqrt(numpy.diag(cov))

def fit(func, x, y, seed=(), fit_range=None, **kwargs):
    if fit_range is not None:
        sel = (fit_range[0] <= x) & (x < fit_range[1])
        x, y = x[sel], y[sel]
        
    vals, cov = scipy.optimize.curve_fit(func, x, y, seed, **kwargs)
    
    fitf = lambda x: func(x, *vals)
    
    return fitf, vals, get_errors(cov)


class unsupervised_eventID(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, head,
                 image_meta,
                 image_key   = "pmaps",
                 lr_scheduler=None):
        super().__init__()

        self.args         = args
        self.encoder      = encoder
        self.head         = head
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler

        self.image_size   = torch.tensor(image_meta['size'][0])
        self.image_origin = torch.tensor(image_meta['origin'][0])

        self.log_keys = ["loss"]

        if args.mode.optimizer.loss_balance_scheme == LossBalanceScheme.even:
            weight = torch.tensor([0.582, 1.417])
        else: weight = None

        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)


    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def forward(self, batch):


        representation = self.encoder(batch)

        # logits = representation
        logits = self.head(representation)
        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        logits = self(image)

        prediction = self.predict_event(logits)
        loss = self.calculate_loss(batch, logits, prediction)

        accuracy_dict = self.calculate_accuracy(prediction, batch['label'])



        metrics = {
            'loss/loss' : loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="train")
        metrics = { "/train/" + key : metrics[key] for key in metrics}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):


        image = batch[self.image_key]

        logits = self(image)


        prediction = self.predict_event(logits)
        loss = self.calculate_loss(batch, logits, prediction)

        accuracy_dict = self.calculate_accuracy(prediction, batch['label'])



        metrics = {
            'loss/loss' : loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="val")
        metrics = { "/val/" + key : metrics[key] for key in metrics}
        self.log_dict(metrics, logger)
        return

    # def prefit_distribution(self, all_energies):
    #     print(all_energies)

    #     # all_energies = torch.tensor(all_energies, dtype=torch.float)

    #     # Define the bins used in the fit:
    #     energy_bins = numpy.arange(1.45, 1.8, 0.005)
    #     e_bin_centers = 0.5*(energy_bins[1:] + energy_bins[:-1])
    #     bin_widths  = energy_bins[1:] - energy_bins[:-1]
        
    #     # Bins used for plotting, if needed:
    #     plotting_bins    = numpy.arange(1.4, 1.8, 0.0005)
    #     plotting_centers = 0.5*(plotting_bins[1:] + plotting_bins[:-1])

    #     from matplotlib import pyplot as plt


    #     # Create a histogram of the data over the fitting bings:
    #     data_events, _ = numpy.histogram(all_energies, energy_bins)
    #     # a0, tau, amp, mu, sigma
    #     seed = (100.0, 1.0, 400.0, 1.65, 0.05)


    #     fit_fn = lambda fit_params : expgauss(e_bin_centers, fit_params) - data_events
    #     residual_fn = lambda fit_params : (fit_fn(fit_params)**2).sum()

    #     ret = scipy.optimize.curve_fit(expgauss, e_bin_centers, data_events, p0=seed)
        
    #     raw_fitf, raw_values, raw_errors = fit(expgauss, e_bin_centers, data_events, seed)
    #     print("Fit values: ", raw_values)
        
    #     # Lastly, we're going to train the network's distribution against method-of-moments

    #     # So, compute via method of moments:
    #     self.min_e = e_bin_centers[0]; self.max_e = e_bin_centers[-1]

    #     # Treat these as discrete probability distributions:
    #     sig_vals = gauss(e_bin_centers, *raw_values[2:])
    #     bkg_vals =   exp(e_bin_centers, *raw_values[0:2])

    #     sig_vals = sig_vals / numpy.sum(sig_vals)
    #     bkg_vals = bkg_vals / numpy.sum(bkg_vals)

    #     # Computing the moments is a computation on the x values, treating the other stuff
    #     # as probability distributions.

    #     # The first moment (aka <x> is the probability times x, summed)

    #     sig_mean = numpy.sum(e_bin_centers * sig_vals)
    #     # print("sig_sum: ", sig_mean)

    #     bkg_mean = numpy.sum(e_bin_centers * bkg_vals)
    #     # print("bkg_mean: ", bkg_mean)

    #     # print("sig_vals: ", sig_vals)
    #     # print("bkg_vals: ", bkg_vals)

    #     n_moments = 5
    #     sig_moments = []
    #     bkg_moments = []
    #     for i in range(0, n_moments):
    #         sig_moments.append( numpy.sum( sig_vals * (e_bin_centers - sig_mean)**i ) )
    #         bkg_moments.append( numpy.sum( bkg_vals * (e_bin_centers - bkg_mean)**i ) )

    #     # print("numpy.sum(sig_vals): ", numpy.sum(sig_vals))
    #     # print("numpy.sum(bkg_vals): ", numpy.sum(bkg_vals))
    #     # print("sig_moments: ", sig_moments)
    #     # print("bkg_moments: ", bkg_moments)

    #     self.sig_moments = sig_moments
    #     self.bkg_moments = bkg_moments
    #     self.sig_mean    = sig_mean
    #     self.bkg_mean    = bkg_mean

    #     # ret = torchmin.least_squares(fit_fn, seed, gtol=None)

    #     # # Plot the data and initial/final fit:
    #     # init_fit = expgauss(plotting_centers, *seed)
    #     # final_fit = expgauss(plotting_centers, *ret[0])
    #     # # print("Final res: ", ret[0])
    #     # plt.bar(e_bin_centers, data_events, width=bin_widths, label="Data", zorder=3)
    #     # plt.plot(plotting_centers, init_fit, label="Seed", color="r", zorder=4)
    #     # plt.plot(plotting_centers, final_fit, label="Fit", color="g", zorder=4)
    #     # plt.grid(True)
    #     # plt.legend()
    #     # plt.savefig("Test_initial_fit.pdf")
    #     # plt.close()



    #     return


    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            message = format_log_message(
                log_keys = self.log_keys,
                metrics  = metrics,
                batch_size = self.args.run.minibatch_size,
                global_step = self.global_step,
                mode = mode
            )

            logger.info(message)

    def exit(self):
        pass

    def unravel_index(self,index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def predict_event(self, prediction):

        batch_size = prediction.shape[0]

        class_prediction = torch.argmax(prediction, dim=-1)

        return class_prediction


    def calculate_accuracy(self, prediction, labels):
        SIGNAL = 1
        BACKGROUND = 0

        accuracy = prediction == labels
        print(labels)
        is_signal     = labels == SIGNAL
        is_background = labels == BACKGROUND

        sig_acc  = torch.mean(accuracy[is_signal].to(torch.float32))
        bkg_acc = torch.mean(accuracy[is_background].to(torch.float32))
        accuracy = torch.mean(accuracy.to(torch.float32))


        return {
            "acc/accuracy" : accuracy,
            "acc/sig_acc": sig_acc,
            "acc/bkg_acc": bkg_acc,
        }


    def calculate_loss(self, batch, logits, prediction=None):
        SIGNAL = 1
        BACKGROUND = 0

        # # Use the energy to compute a weak label:

        # weak_label = BACKGROUND*torch.ones( 
        #     (len(batch['energy']),),
        #     dtype=torch.int64,
        #     device=logits.device
        #     )

        # # Select the weak signal region and set it to 1:
        # sig_region_lower = batch['energy'] > 1.58
        # sig_region_upper = batch['energy'] < 1.62
        # sig_region = torch.logical_and(sig_region_upper, sig_region_lower)
        # weak_label[sig_region] = SIGNAL

        # print(weak_label)
        # print(batch['label'])
        # print(weak_label.device)
        # print(logits.device)

        if self.args.mode.optimizer.loss_balance_scheme == LossBalanceScheme.focal:
            # This section computes the loss via focal loss, since the classes are imbalanced:
            # Create the full label:
            y = torch.nn.functional.one_hot(weak_label, logits.size(-1))
            softmax = torch.nn.functional.softmax(logits) 
            softmax = softmax.clamp(1e-7, 1. - 1e-7)
            loss = - y * torch.log(softmax)

            # Apply the focal part:
            loss = loss * (1 - softmax)**2
            loss = loss.sum(axis=-1).mean()
        else:
            # print(logits.shape)
            # print(batch['label'].shape)
            # print(batch['label'])
            # loss = self.criterion(logits, target = weak_label)
            loss = self.criterion(logits, target = batch['label'])
        print("Loss: ", loss)
        return loss


        # # The loss here is based on method of moments.  We attempt to backprop
        # # the final layer based on the difference between the fitted moments, done
        # # upfront, and the observed moments here.

        # print(logits.shape)
        # print(logits)
        # # print(batch_energy)
        # # Select the signal and bkg events from the logits
        # # Indexing doesn't make a smooth gradient so using a softmax here: (NOT)

        # sigmoid = torch.sigmoid(logits)


        # sig_weighted_energy = batch['energy'] * sigmoid
        # bkg_weighted_energy = batch['energy'] * (1 - sigmoid)

        # sig_weight = torch.sum(sigmoid)
        # bkg_weight = torch.sum(1-sigmoid)

        # sig_mean_e = torch.sum(sig_weighted_energy) / sig_weight
        # bkg_mean_e = torch.sum(bkg_weighted_energy) / bkg_weight

        # # bkg_mean = torch.sum(softmax[:,SIGNAL]*batch['energy'])  / sig_norm
        # # sig_mean = torch.sum(softmax[:,BACKGROUND]*batch['energy']) / bkg_norm

        # print(sig_mean_e)
        # print(bkg_mean_e)

        # sig_moments = []
        # bkg_moments = []
        # # for i in range(n_moments):

        return loss


    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters(), self.args.mode.optimizer.weight_decay)

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]


def create_lightning_module(args, datasets, transforms=None, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)
    image_meta = example_ds.dataset.image_meta(args.data.image_key)
    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks.classification_head import build_networks
    encoder, class_head = build_networks(args, image_shape)

    # For this, we use the augmented data if its available:
    image_key = args.data.image_key
    if args.data.transform1:
        image_key += "_1"
    elif args.data.transform2:
        image_key += "_2"

    model = unsupervised_eventID(
        args,
        encoder,
        class_head,
        image_meta,
        image_key,
        lr_scheduler,
    )
    return model
