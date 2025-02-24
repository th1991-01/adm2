import os
import torch
import numpy as np
from tqdm import tqdm

from dynamics.adm_dynamics import ADMDynamics
from components.static_fns import STATICFUNC
from env.model_as_sim import ModelSim

from .base_trainer import BASETrainer

class OPERunner(BASETrainer):
    """ offline MBRL trainer """

    def __init__(self, args):
        if args.env == "neorl":
            task, data_type, version = tuple(args.env_name.split('-'))
            args.env_name = task + '-' + version
            args.data_type = data_type

        super(OPERunner, self).__init__(args)

        # init dynamics model
        task = args.env_name.split('-')[0]
        if args.env == "neorl": task = "neorl-" + task
        if args.env == "maze": task = task + "-" + args.env_name.split('-')[1]
        self.static_fn = STATICFUNC[task.lower()]
        self.max_adm_step = args.max_adm_step
        if args.dyna_model == "adm":
            self.dyna_model = ADMDynamics(
                obs_dim=np.prod(args.obs_shape),
                action_dim=args.action_dim,
                hidden_dim=args.model_hidden_dim,
                device=args.device
            )

        # lr schedule
        if args.lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.agent.actor_optim, args.n_epochs)
        else:
            self.lr_scheduler = None

        # other parameters
        self.model_lr = args.model_lr
        self.max_adm_step = args.max_adm_step
        self.rollout_batch_size = args.rollout_batch_size
        self.rollout_length = args.rollout_length
        self.warmup_steps = args.warmup_steps
        self.n_epochs = args.n_epochs
        self.step_per_epoch = args.step_per_epoch
        self.updates_per_step = args.updates_per_step
        
    def run(self):
        if self.load_model:
            # load dynamics model
            load_path = os.path.join(self.load_dir, "dyna_seed-{}.pth".format(self.load_seed))
            state_dict = torch.load(load_path)
            self.dyna_model.load_state_dict(state_dict)
        else:
            # learn dynamics model
            self.dyna_model.learn_from(
                max_adm_step=self.max_adm_step,
                buffer=self.dataset,
                lr=self.model_lr,
                batch_size=self.batch_size
            )
            self._save({})
        
        # build model-based env
        init_seqs = self.dataset.sample_all_nstep(self.max_adm_step-1)
        init_seqs["s"] = torch.cat((init_seqs["s"], init_seqs["s_"][:, -1:]), dim=1)
        self.model_env = ModelSim(
            dynamics=self.dyna_model,
            static_fn=self.static_fn,
            max_steps=self.rollout_length,
            init_obs_seqs=init_seqs["s"],
            init_act_seqs=init_seqs["a"],
            n_parallels=self.rollout_batch_size
        )
        
        self._ope()

    def _ope(self):
        """ ope """
        # TODO
