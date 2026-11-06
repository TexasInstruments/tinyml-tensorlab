"""Regression test: radar's modelmaker params default (lr_scheduler) must be
a value init_lr_scheduler actually accepts. Found via manual E2E verification
while wiring compile_model into radar's modelmaker layer -- any radar
training invoked through modelmaker with default params (no lr_scheduler
override) crashed during optimizer/scheduler setup with:
RuntimeError: Invalid lr scheduler 'constantlr'. Only StepLR, CosineAnnealingLR
and ExponentialLR are supported.
'constantlr' isn't one of those three, and isn't the special 'none' value
(which is what actually produces a genuinely constant LR via
ConstantLR(factor=1.0, total_iters=0)) -- it's an unsupported typo/misnaming."""
import torch

from tinyml_tinyverse.common.utils.utils import init_lr_scheduler


def test_radar_default_lr_scheduler_is_accepted_by_init_lr_scheduler():
    from tinyml_modelmaker.ai_modules.radar.params import init_params

    params = init_params()
    lr_scheduler_value = params.training.lr_scheduler

    dummy_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

    # Should not raise. Pre-fix: RuntimeError("Invalid lr scheduler 'constantlr'...")
    scheduler = init_lr_scheduler(
        lr_scheduler=lr_scheduler_value, optimizer=dummy_optimizer, epochs=10, lr_warmup_epochs=0,
    )
    assert scheduler is not None
