import tempfile
import unittest
from pathlib import Path

import _support  # noqa: F401  # puts src/ on sys.path

from accelerate import Accelerator

from _support import tiny_config

from litespecformer.config import LiteSpecFormerTrainingConfig
from litespecformer.trainer import LiteSpecFormerPreTrainer


class LiteSpecFormerPreTrainerTest(unittest.TestCase):
    def test_trainer_builds_a_model_optimizer_and_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train" / "subset0").mkdir(parents=True)
            (root / "test").mkdir()
            training_config = LiteSpecFormerTrainingConfig(
                batch_size=2,
                num_training_steps=100,
                num_warmup_steps=10,
                num_test_steps=20,
                learning_rate=1e-3,
            )
            training_config.checkpoint_path = str(root / "checkpoints")
            training_config.data_path = str(root)

            trainer = LiteSpecFormerPreTrainer(
                accelerator=Accelerator(),
                model_config=tiny_config(),
                training_config=training_config,
            )

            self.assertEqual(trainer.num_training_steps, 100)
            self.assertEqual(trainer.num_training_subset, 1)
            self.assertGreater(len(trainer.get_trainable_params()), 0)
            self.assertGreater(trainer.learning_rate, 0.0)
            self.assertTrue(Path(trainer.checkpoint_dir).is_dir())
            self.assertEqual(trainer.optimizer.__class__.__name__, "AdamW")


if __name__ == "__main__":
    unittest.main()
