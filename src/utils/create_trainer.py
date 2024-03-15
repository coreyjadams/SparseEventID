import torch
import pytorch_lightning as pl

from lightning_fabric.plugins.environments import MPIEnvironment, LightningEnvironment
from pytorch_lightning.callbacks           import ModelCheckpoint, ModelSummary


class OversubscribeMPI(MPIEnvironment):

    def __init__(self, oversubscribe=1):
        super().__init__()
        self.os = oversubscribe
    def local_rank(self):
        lr = super().local_rank()
        return lr // self.os

def create_trainer(args, lightning_model, datasets):

    from src.config.config import Precision

    # Map the precision to lightning args:
    if args.run.precision == Precision.mixed:
        precision = 16
    elif args.run.precision == Precision.bfloat16:
        precision = "bf16"
    else:
        precision = 32

    # Map the profiling to lightning args:
    if args.run.profile:
        profiler = "simple"
    else:
        profiler  = None


    oversubscribe = args.framework.oversubscribe
    if args.run.distributed:
        if oversubscribe == 1:
            environment = MPIEnvironment()
        else:
            environment = OversubscribeMPI(oversubscribe)
    else:
        environment = LightningEnvironment()

    # Distributed strategy:
    if args.run.distributed:
        from src.config.framework import DistributedMode
        if args.framework.distributed_mode == DistributedMode.horovod:
            strategy = "horovod"
        elif args.framework.distributed_mode == DistributedMode.DDP:
            from pytorch_lightning.strategies import DDPStrategy
            backend = "nccl"
            if oversubscribe > 1:
                backend = "gloo"
            strategy = DDPStrategy(
                cluster_environment = environment,
                process_group_backend=backend
            )
        elif args.framework.distributed_mode == DistributedMode.deepspeed:
            strategy = "deepspeed"

        # devices   = int(os.environ['LOCAL_SIZE'])
        # num_nodes = int(os.environ['N_NODES'])
        plugins   = []
        # if args.run.compute_mode == ComputeMode.CUDA:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
        #     devices=1
    else:
        from pytorch_lightning.strategies import SingleDeviceStrategy
        plugins   = []
        strategy  = SingleDeviceStrategy(f"{args.run.compute_mode.name.lower()}:0")
        devices   = 1
        num_nodes = 1

    # Configure the logger:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_logger = TensorBoardLogger(
        save_dir = args.output_dir,
        version  = 0,
    )

    checkpoint_dir = args.output_dir + "/checkpoints/"
    model_checkpoint = ModelCheckpoint(
        dirpath = checkpoint_dir,
        every_n_train_steps = 50,
    )


    checkpoint_path = None

    # Checkpoint loading.  First, do we have a path specified?
    if args.mode.weights_location != "":
        if args.mode.restore_encoder_only:
            # In this situation, we only load the encoder
            state_dict = torch.load(args.mode.weights_location)

            encoder_dict = {
                key.replace("encoder.","") : state_dict["state_dict"][key]
                for key in state_dict["state_dict"] if "encoder" in key
            }

            lightning_model.encoder.load_state_dict(encoder_dict)
            # We also FREEZE the encoder:
            for param in lightning_model.encoder.parameters():
                param.requires_grad = False
        else:
            # lightning_model.load_from_checkpoint(args.mode.weights_location)
            checkpoint_path = args.mode.weights_location
    else:
        import glob
        # Check to see if there are already checkpoints present:
        checkpoint_options = glob.glob(checkpoint_dir + "*.ckpt")
        if len(checkpoint_options) > 0:
            checkpoint_path = checkpoint_options[0]
            # print(f"checkpoint_options: {checkpoint_options}")
            # state_dict = lightning_model.load_from_checkpoint(checkpoint_options[0])
            # print("Loaded model from checkpoint")


    # # If we're doing unsupervised training, we have to fit the datasets initially based on energy:
    # if args.name == "unsupervised_eventID":
    #     lightning_model.prefit_distribution(datasets["train"].dataset.ds.energy)

    if 'optimizer' in args.mode:
        accum_grad_batches = args.mode.optimizer.gradient_accumulation
        limit_val_batches = 1
    else:
        accum_grad_batches = 1
        limit_val_batches = 60

    trainer = pl.Trainer(
        accelerator             = args.run.compute_mode.name.lower(),
        default_root_dir        = args.output_dir,
        precision               = precision,
        profiler                = profiler,
        strategy                = strategy,
        enable_progress_bar     = False,
        logger                  = tb_logger,
        log_every_n_steps       = 1,
        max_epochs              = args.run.length,
        # plugins                 = plugins,
        accumulate_grad_batches = accum_grad_batches,
        val_check_interval      = 10,
        check_val_every_n_epoch = None,
        limit_val_batches       = limit_val_batches,
        callbacks               = [model_checkpoint, ModelSummary(max_depth=3,)],
    )

    return trainer, lightning_model, checkpoint_path

    # Try to load the model from a checkpoint:

    # print(trainer.global_step)
    # exit()


    # model_checkpoint.format_checkpoint_name()

    # lightning_model.load_from_checkpoint(args.output_dir + "/checkpoints/")

