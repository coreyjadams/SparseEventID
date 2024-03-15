import torch
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src import logging

logger = logging.getLogger("NEXT")

class rep_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, head, transforms,
                 image_key   = "pmaps",
                 lr_scheduler=None ):
        super().__init__()

        self.args         = args
        self.encoder      = encoder
        self.head         = head
        self.transforms   = transforms
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler
        # self.loss_calc    = loss

        self.log_keys = ["loss"]

        self.save_hyperparameters(ignore=["encoder","head","loss"])

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    # def on_train_batch_start(self, batch, batch_idx):
    #     if self.global_step > 150: return -1

    def forward(self, augmented_images):

        # print(batch.keys())
        #
        # IMPLEMENT LARS
        # CHECK THROUGHPUT WITH PROFILER
        # CHECK MPS SCALE OUT?
        # Launch large scale runs


        representation = [ self.encoder(ad) for ad in augmented_images ]

        logits = [ self.head(r) for r in representation]

        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        augmented_images = [batch[k] for k in self.transforms]

        encoded_images = self(augmented_images)

        # loss = self.loss_calc(encoded_images[0], encoded_images[1])
        # print("Official loss: ", loss, flush=True)
        loss, loss_metrics = self.calculate_loss(encoded_images[0], encoded_images[1])
        # print("my loss: ", loss, flush=True)

        metrics = {
            'opt/loss'        : loss,
            'opt/lr'          : self.optimizers().state_dict()['param_groups'][0]['lr'],
            'opt/alignment'   : loss_metrics["alignment"],
            'opt/log_sum_exp' : loss_metrics["log_sum_exp"],
            'acc/top1'        : loss_metrics["top1"],
            'acc/top5'        : loss_metrics["top5"],
        }

        # self.log()
        self.print_log(metrics, mode="train")
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):

        augmented_images = [batch[k] for k in self.transforms]

        encoded_images = self(augmented_images)

        # loss = self.loss_calc(encoded_images[0], encoded_images[1])
        # print("Official loss: ", loss, flush=True)
        loss, loss_metrics = self.calculate_loss(encoded_images[0], encoded_images[1])
        # print("my loss: ", loss, flush=True)

        metrics = {
            'opt/loss'        : loss,
            'opt/alignment'   : loss_metrics["alignment"],
            'opt/log_sum_exp' : loss_metrics["log_sum_exp"],
            'acc/top1'        : loss_metrics["top1"],
            'acc/top5'        : loss_metrics["top5"],
        }

        # self.log()
        self.print_log(metrics, mode="val")
        # self.log_dict(metrics)
        return loss


    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            message = format_log_message(
                self.log_keys,
                metrics,
                self.args.run.minibatch_size,
                self.global_step,
                mode
            )

            logger.info(message)

    def exit(self):
        pass
#
    def calculate_loss(self, first_images, second_images, temperature=0.1):
        # Each image is represented with k parameters,
        # Assume the batch size is N, so the
        # inputs have shape (N, k)

        # These are pre-distributed shapes:
        N = first_images.shape[0]
        k = first_images.shape[1]

        # Need to dig in here to fix the loss function:
        # https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

        # Also it has LARS to use.


        first_images = first_images / torch.norm(first_images,dim=1).reshape((-1,1))
        second_images = second_images / torch.norm(second_images,dim=1).reshape((-1,1))

        # Take the two tuples, and concatenate them.
        # Then, reshape into Y = (1, 2N, k) and Z = (2N, 1, k)

        c = torch.concat([first_images, second_images], dim=0)

        # Gather all the c up if the world size > 1:

        # All gather turns out to be a differentiable operation in pt_lightning
        # Probably because of this loss function!
        gathered_c = self.all_gather(c)
        gathered_c = gathered_c.reshape((-1, first_images.shape[-1]))


        # Each rank computes only a slice of the global loss matrix, or
        # the memory usage gets out of control.

        # We calculate the dot product between the local and global tensors:
        local_reps = c.reshape((c.shape[0], 1, c.shape[1]))
        all_reps   = gathered_c.reshape((1, gathered_c.shape[0], gathered_c.shape[1]))


        # Assume we have n images per rank, for N global images with N = n * world_size
        # Compute the product of these tensors, which gives shape
        # (2n, 2N, k)
        mat =  local_reps*all_reps

        # We need to compute the function (sim(x,y)) for each element in the 2N sequent.
        # Since the are normalized, we're computing x^T . Y / (||x||*||y||),
        # but the norms are equal to 1.
        # So, summing the matrix over the dim = 0 and dim = 1 computes this for each pair.

        sim = torch.sum(mat, dim=-1) / temperature



        # Now, sim is of shape [2*n, 2*N]

        # This yields a symmetric matrix, diagonal entries equal 1.  Off diagonal are symmetrics and < 1.

        # sim = torch.exp(sim / temperature)
        # Now, for every entry i in C (concat of both batches), the sum of sim[i] - sim[i][i] is the denominator

        device = sim.device

        # Since we have a non-symmetric matrix, need to build a non-symmetric index:
        positive = torch.zeros(sim.shape, device=device)

        # We concatenated all the local examples, and compute symmetric positive pairs
        # So for the first N entries, the index of the positive pair is i + N  (locally)
        # For the second N entries, the index of the positive pair is i - N (locally)
        # with a distributed run, we've squashed all the similarity scores together.
        # to a shape of [2*N, 2*N*Size]
        # Each 2*N by 2*N block is the local positive indexes, all others are negative.
        # That means that the index is shifted by global_rank*2*N

        access_index_x = torch.arange(2*N)
        # For the first N, the y-index is equal to x + 2*N
        # For the second N
        access_index_y = torch.arange(2*N)
        # Shift by +/- N:
        access_index_y[0:N] = access_index_y[0:N] + N
        access_index_y[N:]  = access_index_y[N:] - N

        access_index_y +=  self.global_rank * 2*N

        # print("access_index_y: ", access_index_y, flush=True)

        positive[access_index_x, access_index_y] = 1

        # For the negative, we invert the positive and have to 0 out the self-index entries
        negative = 1 - positive
        # access_index_y = access_index_x + self.global_rank*2*N
        # negative[access_index_x, access_index_y] = 0.

        # THESE WORK IF IT'S NOT DISTRIBUTED
        # positive = torch.tile(torch.eye(N, device=device), (2,2))
        # # Unsure if this line is needed?
        # positive = positive - torch.eye(2*N, device=device)
        #
        # negative = - (torch.eye(2*N, device=device) - 1)

        with torch.no_grad():
            # Here, we can compute the top-k metrics for this batch, since we have the global state:
            # We want the top 5 entries but the self-sim is obviously perfect.
            # So take the top 6 and reject the first.
            topk = torch.topk(sim, k=6, dim=-1, sorted=True)

            # Top 1 is just an equality check:
            top1_acc = topk.indices[:,1] == access_index_y.to(topk.indices.device)
            top1_acc = torch.mean(top1_acc.to(torch.float))
          
            # Top 5 is a little more complicated:
            # Compute the index distance to the correct index, abs value:
            top5_acc_dist = torch.abs(topk.indices[:,1:] - access_index_y.to(topk.indices.device).reshape(-1,1))
            # Get the minumum value, and see if it is less than 5:
            min_values, _ = torch.min(top5_acc_dist, dim=-1)
            top5_acc =  min_values < 5.
            # Average over the batch dimension:
            top5_acc = torch.mean(top5_acc.to(torch.float))


        negative_examples = sim * negative
        positive_examples = sim * positive

        # Now, positive/negative examples is the temperature normalized similarity.
        # we need to sum across the whole batch dimension to compute it per-example:


        # Compute the alignment, summed over the entire global batch:
        alignment = torch.sum(positive_examples, dim=-1)

        # Compute the exp, which we'll eventually sum and log:
        exp = torch.sum(torch.exp(negative_examples), dim=-1)

        # print("Alignment: ", alignment, flush=True)
        # print("exp: ",       exp, flush=True)


        # And compute the logsumexp of the negative examples:
        log_sum_exp = torch.log(exp )


        # Additionally, we can compute the "floor" of the loss at this batch size:
        # floor = torch.log(1.*N) - 1.

        loss_metrics = {
            "alignment"   : torch.mean(alignment),
            "log_sum_exp" : torch.mean(log_sum_exp),
            "top1"        : top1_acc,
            "top5"        : top5_acc,
            # "floor"       : floor,
        }

        loss = torch.mean( - alignment + log_sum_exp)
        return loss, loss_metrics

    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters(), self.args.mode.optimizer.weight_decay)

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]

def create_lightning_module(args, datasets, transforms, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)
    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks import classification_head
    encoder, classification_head = classification_head.build_networks(args, image_shape)

    from src.networks import NT_Xent
    local_size = int(args.run.minibatch_size / args.run.world_size)
    # loss = NT_Xent.NT_Xent(batch_size = local_size, temperature=0.1,
        # world_size=args.run.world_size)

    model = rep_trainer(
        args,
        encoder,
        classification_head,
        transforms,
        # loss,
        args.data.image_key,
        lr_scheduler
    )
    return model
