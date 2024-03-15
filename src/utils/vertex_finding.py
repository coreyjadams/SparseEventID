import sys, os

import torch
import pytorch_lightning as pl

# torch.set_float32_matmul_precision('high')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src import logging
from src.config.mode import ModeKind

logger = logging.getLogger("NEXT")

class vertex_learning(pl.LightningModule):
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

        if self.args.mode.name == ModeKind.inference: 
            self.metrics_list = None


    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def forward(self, batch):



        representation = self.encoder(batch)
        logits = self.head(representation)


        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        prediction = self(image)

        reference_shape = prediction.shape[2:]
        vertex_labels = self.compute_vertex_labels(batch, reference_shape)

        prediction_dict = self.predict_event(prediction)

        anchor_loss, regression_loss, event_loss = self.vertex_loss(vertex_labels, prediction)

        loss = anchor_loss + 0.1*regression_loss


        accuracy_dict = self.calculate_accuracy(prediction_dict, batch, vertex_labels)

        metrics = {
            'loss/loss' : loss,
            'loss/anchor_loss' : anchor_loss,
            'loss/regression_loss' : regression_loss,
            'loss/event_loss'  : event_loss,
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

        prediction = self(image)

        reference_shape = prediction.shape[2:]
        vertex_labels = self.compute_vertex_labels(batch, reference_shape)

        prediction_dict = self.predict_event(prediction)

        # In inference mode, intercept the prediction and store it in a dictionary:
        if hasattr(self, "metrics_list") and self.metrics_list is None:
            self.metrics_list = {}
            # for key in vertex_labels.keys():
                # self.metrics_list[key] = []
            for key in ["label", 'vertex_true', 'entries', 'energy']:
                self.metrics_list[key] = []
            for key in prediction_dict.keys():
                self.metrics_list[key] = []


            # Construct the save vector:
            for key in ["label", 'vertex_true', 'entries', 'energy']:
                if key == "vertex_true":
                    self.metrics_list["vertex_true"].append(batch["vertex"])
                else:
                    self.metrics_list[key].append(batch[key])

            for key in prediction_dict.keys():
                self.metrics_list[key].append(prediction_dict[key])

        anchor_loss, regression_loss, event_loss = self.vertex_loss(vertex_labels, prediction)

        loss = anchor_loss + 0.1*regression_loss + 0.1*event_loss


        accuracy_dict = self.calculate_accuracy(prediction_dict, batch, vertex_labels)

        metrics = {
            'loss/loss' : loss,
            'loss/anchor_loss' : anchor_loss,
            'loss/regression_loss' : regression_loss,
            'loss/event_loss'  : event_loss,
            # 'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="val")
        metrics = { "/val/" + key : metrics[key] for key in metrics}
        self.log_dict(metrics, logger)
        return

    def on_validation_end(self) -> None:

        # for key in self.metrics_list.keys():
        #     print(key)
        #     print(len(self.metrics_list[key]))
        #     print(self.metrics_list[key][0].shape)
        if hasattr(self, "metrics_list"):
            self.metrics_list = {
                key : torch.concat(self.metrics_list[key], axis=0).cpu().numpy()
                for key in self.metrics_list.keys()
            }
            print(self.metrics_list.keys())
            for key in self.metrics_list.keys():
                print(key, self.metrics_list[key].shape)

            fname = self.args.output_dir
            fname += "/validation_output/"
            if self.global_rank == 0:
                os.makedirs(fname, exist_ok=True)
            fname += f"val_rank_{self.global_rank}"
            import numpy
            numpy.savez(fname, **self.metrics_list)

            
        return super().on_validation_end()

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
        image_size = prediction.shape[2:]
        
        # Select the prediction slices:
        anchor_prediction     = prediction[:,0]
        label_prediction     = prediction[:,1]
        regression_prediction = prediction[:,2:]

        # Pick the anchor with the highest score:
        _, max_index = torch.max(
            torch.reshape(anchor_prediction, (batch_size, -1)), dim=1)


        # Declare it a signal event if it's more than 0.5 for a vertex:


        prediction_dict = {
            "anchor" : anchor_prediction,
        }


        # Take the maximum location and use that to infer the vertex prediction and class label.
        # Need to unravel the index since it's flattened ...
        vertex_anchor = self.unravel_index(max_index, image_size)

        batch_index = torch.arange(batch_size)
        selected_boxes = regression_prediction[batch_index,:,vertex_anchor[0], vertex_anchor[1],vertex_anchor[2]]
        selected_label = label_prediction[batch_index, vertex_anchor[0], vertex_anchor[1], vertex_anchor[2]]

        # Convert the selected anchors + regression coordinates into a vertex in 3D:
        vertex_anchor = torch.stack(vertex_anchor,dim=1)
        vertex = self.image_origin + (vertex_anchor+selected_boxes)*self.anchor_size

        prediction_dict['vertex'] = vertex
        prediction_dict['label']  = (selected_label > 0.5).to(torch.int32)

        return prediction_dict

    def calculate_accuracy(self, prediction, batch, vertex_labels):


        event_accuracy = (prediction['label'] == batch['label']).to(torch.float32)

        is_signal = batch['label'] == 1

        acc_sig = torch.mean(event_accuracy[is_signal])
        acc_bkg = torch.mean(event_accuracy[torch.logical_not(is_signal)])

        # Reduce the global event accuracy:
        event_accuracy = torch.mean(event_accuracy,dim=0)

        vertex_truth = batch["vertex"]
        vertex_pred  = prediction["vertex"]

        vertex_resolution = vertex_pred - vertex_truth



        # How often is the vertex detected in the right anchor?
        batch_size = batch['label'].shape[0]
        _, true_vertex_loc = torch.max(vertex_labels["anchor"].reshape((batch_size,-1)), dim=1)
        _, pred_vertex_loc = torch.max(prediction   ["anchor"].reshape((batch_size,-1)), dim=1)


        vertex_anchor_acc = true_vertex_loc == pred_vertex_loc
        vertex_anchor_acc = torch.mean(vertex_anchor_acc.to(torch.float32))

        vertex_displacement = torch.sqrt(torch.sum(vertex_resolution**2, dim=1))
        vtx_5 = (vertex_displacement < 5.0).to(torch.float32)
        vtx_10 = (vertex_displacement < 10.0).to(torch.float32)
        vtx_20 = (vertex_displacement < 20.0).to(torch.float32)

        return {
            "acc/accuracy" : event_accuracy,
            "acc/sig_acc"  : acc_sig,
            "acc/bkg_acc"  : acc_bkg,
            "acc/vertex_decection" : vertex_anchor_acc,
            "acc/vertex_x" : torch.mean(vertex_resolution[:,0]),
            "acc/vertex_y" : torch.mean(vertex_resolution[:,1]),
            "acc/vertex_z" : torch.mean(vertex_resolution[:,2]),
            "acc/vertex"   : torch.mean(vertex_displacement),
            "acc/vertex_5" : torch.mean(vtx_5),
            "acc/vertex_10": torch.mean(vtx_10),
            "acc/vertex_20": torch.mean(vtx_20),

        }


    def compute_vertex_labels(self, batch, reference_shape):

        # Basic properties infered from the input:
        vertex_label = batch['vertex']
        target_device = vertex_label.device
        batch_size = vertex_label.shape[0]

        # Info for determining image size:
        self.image_size   = self.image_size.to(target_device, dtype=torch.float32)
        self.image_origin = self.image_origin.to(target_device, dtype=torch.float32)

        # the size of each anchor box can be found first:
        if not hasattr(self, "anchor_size"):
            labels_spatial_size = torch.tensor(reference_shape).to(target_device)
            self.anchor_size = torch.tensor(self.image_size / labels_spatial_size)

        # 
        class_shape = (batch_size,) + reference_shape
        regression_shape = (batch_size, 3,) + reference_shape

        # Start with the reference shape to define the labels
        regression_labels = vertex_label.to(target_device)

        # The first goal is to figure out which binary labels to turn on, if any.
        # We take the 3D vertex location and figure out which pixel it aligns with in X/Y/Z.

        # Identify where the vertex is with the 0,0,0 index at 0,0,0 coordinate::
        relative_vertex = regression_labels - self.image_origin

        # Map the relative location onto the anchor grid
        anchor_index = (relative_vertex // self.anchor_size).to(torch.int64)


        # Select the target anchor indexes:
        batch_index = torch.arange(batch_size)

 
        # Now create the anchor labels, mostly zero:
        anchor_labels = torch.zeros(size = class_shape, device=target_device, dtype=torch.float32)
        anchor_labels[batch_index, anchor_index[:,0], anchor_index[:,1], anchor_index[:,2] ]  = 1


        # Set the class labels
        # This is outdated since most events have a vertex now
        # class_labels[active_anchors,target_anchors[:,0],target_anchors[:,1],target_anchors[:,2]] = 1.

        # We need to map the regression label to a relative point in the anchor window.
        anchor_start_point = self.anchor_size * anchor_index + self.image_origin
        regression_labels = (regression_labels - anchor_start_point) / self.anchor_size

    
        regression_labels = regression_labels.reshape(regression_labels.shape + (1,1,1,))

        # Lastly, we do an event-wide weight based on the presence of a vertex:
        weight = 0.5*torch.ones((batch_size,), device=target_device)
        is_signal = batch['label'] == 1
        weight[is_signal] = 5

        

        return {
            "class"      : batch['label'],
            "anchor"     : anchor_labels,
            "regression" : regression_labels,
            "weight"     : weight
        }

    def vertex_loss(self, vertex_labels, prediction):


        anchor_prediction = prediction[:,0,:,:,:]
        anchor_labels = vertex_labels["anchor"]
            

        # print("Anchor prediction: ", anchor_prediction.shape)
        # print("anchor_prediction[0]: ", anchor_prediction[0])
        # print("anchor_labels[0]: ", anchor_labels[0])
        # Compute the loss:
        anchor_loss = torch.nn.functional.binary_cross_entropy(
            anchor_prediction, anchor_labels, reduction="none")

        # print(anchor_loss.shape)
        # print("anchor_loss[0]: ", anchor_loss[0])


        focus = (anchor_prediction - anchor_labels)**2

        # print("focus.shape: ", focus.shape)
        # print("focus[0]: ", focus[0])

        # Sum over images; average over batch dimensions
        batch_size = anchor_loss.shape[0]
        anchor_loss = torch.reshape(anchor_loss*focus, (batch_size,-1))
        

        # print("Final anchor_loss.shape: ", anchor_loss.shape)
        # print("Final anchor_loss[0]: ", anchor_loss[0])
        
        # Sum over entire images, but not the batch:
        # anchor_loss = torch.sum(anchor_loss, dim=1)
        # exit()
        anchor_loss = torch.mean(anchor_loss)

        # For the regression loss, it's actually easy to calculate.
        # Compute the difference between the regression point and the prediction point
        # on every anchor, and then scale by the label for that anchor (present/not-present)

        regression_prediction = prediction[:,2:,:,:,:]

        regression_loss = (vertex_labels['regression'] - regression_prediction)**2

        target_shape = regression_loss.shape
        # Scale, but put a 1 in the shape to broadcast over x/y/z
        target_shape = (target_shape[0],1) + target_shape[2:]
        regression_loss = regression_loss * anchor_labels.reshape((target_shape))



        # Sum over all anchors (most are 0) and batches
        regression_loss = torch.sum(regression_loss)
        weight = torch.sum(anchor_labels) + 0.0001

        # Finally, for the label loss, we can compute it like regression:
        # We compute for every box, and scale by the anchor value

        event_prediction = prediction[:,1,:,:,:].detach()
        # Add 3 dimensions to the label to broadcast:
        event_label = vertex_labels['class'].reshape((-1,1,1,1))

        # Compute the binary cross entropy directly, sigmoid already applied:
        event_loss = - event_label * torch.log(event_prediction)


        event_loss = event_loss * anchor_labels
        event_loss = torch.sum(event_loss)
        return anchor_loss, regression_loss / weight, event_loss / weight


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
    from src.networks.yolo_head import build_networks
    encoder, yolo_head = build_networks(args, image_shape)

    model = vertex_learning(
        args,
        encoder,
        yolo_head,
        image_meta,
        args.data.image_key,
        lr_scheduler,
    )
    return model
