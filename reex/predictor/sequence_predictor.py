from typing import List, Dict, Union, Iterable

import numpy
import torch

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from overrides import overrides


@Predictor.register("relation_classifier", exist_ok=True)
class RelationClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.
    Registered as a `Predictor` with name "text_classifier".
    """

    def predict(self, tokens: List[str], relation: Iterable[Union[str, int]]) -> JsonDict:
        return self.predict_json({"tokens": tokens, "relation": relation})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        tokens = json_dict["tokens"]
        relation = json_dict["relation"]
        return self._dataset_reader.text_to_instance(tokens, relation)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        predicted_label = outputs["answers"]
        new_instance.add_field("label", LabelField(int(predicted_label), skip_indexing=True))
        return [new_instance]

    @overrides
    def get_interpretable_layer(self) -> torch.nn.Module:
        """
        Returns the input/embedding layer of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct input/embedding layer.
        For the cases where the input layer _is_ an embedding layer, this should be the
        layer 0 of the embedder.
        """
        return self._model.embeddings

    @overrides
    def get_interpretable_text_field_embedder(self) -> torch.nn.Module:
        """
        Returns the input/embedding layer of the model.
        If the predictor wraps around a non-AllenNLP model,
        this function should be overridden to specify the correct input/embedding layer.
        For the cases where the input layer _is_ an embedding layer, this should be the
        layer 0 of the embedder.
        """
        return self._model.embeddings

    def __hash__(self):
        return hash(self._model) ^ hash(self._dataset_reader)

