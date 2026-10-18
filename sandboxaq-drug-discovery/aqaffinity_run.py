"""Run the aqaffinity CLI with two compatibility shims. Usage: same arguments as `aqaffinity`.

1. AQAffinity's affinity stage calls trainer.predict a second time, which re-runs
   the data module's prepare_data and tries to create the ColabFold MSA workspace
   again; openfold3 0.5 creates it with exist_ok=False and aborts. The existing
   workspace is allowed (and the runner YAML should set
   msa_computation_settings.msa_output_directory for the same reason).
2. The affinity head hard-codes use_deepspeed_evo_attention=True, which needs a
   DeepSpeed build with CUDA ops. The flag only selects a memory-efficient kernel
   for the same attention, so the head is run with OpenFold3's default attention.
"""
import sys
from openfold3.core.data.tools import colabfold_msa_server as c

for name in dir(c):
    cls = getattr(c, name)
    if isinstance(cls, type) and hasattr(cls, "create_workspace") and hasattr(cls, "workspace_directory"):
        def create_workspace(self):
            self.workspace_directory.mkdir(parents=True, exist_ok=True)
            self._workspace_created = True
        cls.create_workspace = create_workspace

import torch
from aqaffinity.modules.model.runner import BindingAffinityPredictor

def forward(self, input_data):
    outs = []
    for m in (self.model_1, self.model_2):
        outs.append(m(input_data["batch"], input_data["outputs"]["si_input"], input_data["outputs"],
                      chunk_size=self.config.chunk_size, use_deepspeed_evo_attention=False, use_lma=False,
                      inplace_safe=not torch.is_grad_enabled(), _mask_trans=True))
    return tuple(outs)
BindingAffinityPredictor.forward = forward

from aqaffinity.run import main
sys.exit(main())
