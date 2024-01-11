import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # create a dictionary params for train conf
        params = {
                "batch_size": 2,
                "epochs": 1,
                "dense_dim": 64,
                "min_samples_per_label": 10,
                "verbose": 1
        }

        # we create a temporary file to store artefacts
        #with tempfile.TemporaryDirectory() as model_dir:
            # run a training
        #   accuracy, _ = run.train("fake",params,model_path=model_dir,add_timestamp=True)
        model = TextPredictionModel.from_artefacts("train/data/artefacts/test/2023-01-06-11-01-52")
        predictions = model.predict(["Is it possible to execute the procedure of a function in the scope of the caller?"],
        top_k=1)
        # assert that accuracy is equal to 1.0
        self.assertEqual(predictions,["php"])

