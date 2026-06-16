from .diffusion import Trainer as DiffusionTrainer
from .gan import Trainer as GANTrainer
from .ode import Trainer as ODETrainer
from .distillation import Trainer as ScoreDistillationTrainer
from .wan22_distillation import Trainer as Wan22ScoreDistillationTrainer
from .wan22fun_distillation import Trainer as Wan22FunScoreDistillationTrainer
from .wan22r_distillation import Trainer as Wan22RScoreDistillationTrainer
__all__ = [
    "DiffusionTrainer",
    "GANTrainer",
    "ODETrainer",
    "ScoreDistillationTrainer",
    "Wan22ScoreDistillationTrainer",
    "Wan22FunScoreDistillationTrainer",
    "Wan22RScoreDistillationTrainer",
]
