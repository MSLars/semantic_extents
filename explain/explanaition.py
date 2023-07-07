import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import srsly
from allennlp.common.util import import_module_and_submodules
from allennlp.interpret.saliency_interpreters import SimpleGradient, IntegratedGradient, \
    SmoothGradient
from cached_path import cached_path
from pydantic import BaseModel
from tqdm import tqdm

import pretrained_models
from explain.input_reduction import InputReduction
from human_reduction.annotation_task_datamodel import HIRAnnotationSample, DataSample
from reex.predictor.sequence_predictor import RelationClassifierPredictor


class DecisionConfidence(BaseModel):
    prediction: str
    prediction_idx: int
    confidence: float
    probabilities: dict[str, float]
    gold_label: str
    gold_label_idx: int

    model_tokens: list[str] = None
    pieces2model_tokens: list[int] = None

    @property
    def is_correct(self):
        return self.prediction == self.gold_label


class Explanation(BaseModel):
    sample: DataSample

    full_confidence: DecisionConfidence
    extended_confidence: DecisionConfidence
    reduced_confidence: DecisionConfidence = None

    saliency_interpretations: dict[str, list[float]]
    indices_in_reduction: list[int] = None

    candidate_area_confidences: list[DecisionConfidence] = None
    candidate_outside_shuffled_confidences: list[DecisionConfidence] = None
    candidate_complete_shuffled_confidences: list[DecisionConfidence] = None

    hir_samples: list[HIRAnnotationSample] = None

    @property
    def semantic_extent(self):
        candidates = [c for c in self.candidate_area_confidences if c.prediction == self.full_confidence.prediction]

        if len(candidates) == 0:
            return self.candidate_area_confidences[-1].model_tokens

        return candidates[0].model_tokens

    def to_json(self):
        return {
            "sample": self.sample.to_json(),
            "full_confidence": self.full_confidence.dict(),
            "extended_confidence": self.extended_confidence.dict(),
            "reduced_confidence": self.reduced_confidence.dict() if self.reduced_confidence else None,
            "saliency_interpretations": self.saliency_interpretations,
            "indices_in_reduction": self.indices_in_reduction,
            "candidate_area_confidences": [c.dict() for c in self.candidate_area_confidences],
            "candidate_outside_shuffled_confidences": [c.dict() for c in
                                                       self.candidate_outside_shuffled_confidences],
            "candidate_complete_shuffled_confidences": [c.dict() for c in
                                                        self.candidate_complete_shuffled_confidences],
            "hir_samples": [s.to_json() for s in self.hir_samples],
        }

    @staticmethod
    def from_json(json_elem):
        return Explanation(sample=DataSample.from_json(json_elem["sample"]),
                           full_confidence=DecisionConfidence(**json_elem["full_confidence"]),
                           extended_confidence=DecisionConfidence(
                               **json_elem["extended_confidence"]),
                           reduced_confidence=DecisionConfidence(
                               **json_elem["reduced_confidence"]) if json_elem[
                               "reduced_confidence"] else None,
                           saliency_interpretations=json_elem["saliency_interpretations"],
                           indices_in_reduction=json_elem["indices_in_reduction"],
                           candidate_area_confidences=[DecisionConfidence(**c) for c in
                                                       json_elem["candidate_area_confidences"]],
                           candidate_outside_shuffled_confidences=[DecisionConfidence(**c) for c in
                                                                   json_elem[
                                                                       "candidate_outside_shuffled_confidences"]],
                           candidate_complete_shuffled_confidences=[DecisionConfidence(**c) for c in
                                                                    json_elem[
                                                                        "candidate_complete_shuffled_confidences"]],
                           hir_samples=[HIRAnnotationSample.from_json(s) for s in
                                        json_elem["hir_samples"]] if "hir_samples" in json_elem else None,
                           )


class Explainer:

    def __init__(self,
                 predictor: RelationClassifierPredictor,
                 include_saliency: bool = True,
                 include_reducer: bool = True,
                 include_rule_based_reduction: bool = True,
                 beam_size: int = 5, ):
        self.predictor = predictor
        self.include_saliency = include_saliency
        self.include_reducer = include_reducer

        self.idx2label = self.predictor._model.label_tokens
        self.label2idx = {v: k for (k, v) in self.idx2label.items()}

        self.saliency_interpreters = {}
        if self.include_saliency:
            self.saliency_interpreters = {
                "simple": SimpleGradient(self.predictor),
                "integrated": IntegratedGradient(self.predictor),
                "smooth": SmoothGradient(self.predictor)
            }

        self.reducer = None
        if self.include_reducer:
            self.reducer = InputReduction(self.predictor, beam_size)

        self.include_rule_based_reduction = include_rule_based_reduction

    def _get_probs_from_logits(self, logits):
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return {v: probs[k] for (k, v) in self.predictor._model.label_tokens.items()}, max(
            probs), np.argmax(probs)

    def _calculate_confidence(self, tokens, relation):
        prediction = self.predictor.predict(tokens=tokens, relation=relation)

        label2prob, max_prob, _ = self._get_probs_from_logits(prediction["logits"])

        gold_label_idx = self.label2idx[relation[4]] if len(relation) > 4 else None
        gold_label = relation[4] if len(relation) > 4 else None

        pieces2model_tokens = [i if i != "None" else -1 for i in prediction["metadata"]["word_ids"]]

        return DecisionConfidence(
            prediction=prediction["predicted_labels"],
            prediction_idx=prediction["answers"],
            confidence=max_prob,
            probabilities=label2prob,
            gold_label_idx=gold_label_idx if gold_label_idx is not None else -1,
            gold_label=gold_label if gold_label is not None else "",
            model_tokens=prediction["metadata"]["model_tokens"],
            pieces2model_tokens=pieces2model_tokens
        )

    def explain(self, sample: DataSample):

        full_confidence = self._calculate_confidence(sample.tokens, sample.relation)

        extend_tokens = sample.tokens[sample.extend[0]: sample.extend[1]]

        extend_relation = [
            sample.relation[0] - sample.extend[0],
            sample.relation[1] - sample.extend[0],
            sample.relation[2] - sample.extend[0],
            sample.relation[3] - sample.extend[0],
            sample.relation[4]
        ]

        extended_confidence = self._calculate_confidence(extend_tokens, extend_relation)

        indices_in_reduction = None
        reduced_confidence = None
        if self.include_reducer:
            reduced = self.reducer.attack_from_json({"tokens": sample.tokens,
                                                     "relation": sample.relation[:4]},
                                                    "tokens",
                                                    "grad_input_1",
                                                    ignore_tokens=[0, 2])

            label2prob, max_prob, predicted_index = self._get_probs_from_logits(
                reduced["logits"][0])
            predicted_label = self.idx2label[predicted_index]

            reduced_confidence = DecisionConfidence(
                prediction=predicted_label,
                prediction_idx=predicted_index,
                confidence=max_prob,
                probabilities=label2prob,
                gold_label_idx=full_confidence.gold_label_idx,
                gold_label=full_confidence.gold_label,
            )

            indices_in_reduction = reduced["indices"][0]

        saliency_interpretations = {}

        if self.include_saliency:
            saliency_interpretations = {}
            for name, interpreter in self.saliency_interpreters.items():
                interpretation = interpreter.saliency_interpret_from_json({"tokens": sample.tokens,
                                                                           "relation": sample.relation[
                                                                                       :4]}, )

                saliency_interpretations[name] = interpretation["instance_1"]["grad_input_1"]

        if self.include_rule_based_reduction:
            candidate_area_confidences = []
            candidate_outside_shuffled_confidences = []
            candidate_complete_shuffled_confidences = []

            try:
                from human_reduction.spacy_annotation_task import create_sample_iterator
                hir_samples = create_sample_iterator(sample)

                for hir_sample in hir_samples:
                    reduced_confidence = self._calculate_confidence(hir_sample.tokens,
                                                                    hir_sample.relation)

                    candidate_area_confidences.append(reduced_confidence)

                    # shuffle all other tokens _ outside of the and the proposed DA

                    outside_shuffle_hir_sample = hir_sample.outside_shuffle(
                        orig_tokens=sample.tokens,
                        orig_relation=sample.relation)

                    outside_shuffle_confidence = self._calculate_confidence(
                        outside_shuffle_hir_sample.tokens,
                        outside_shuffle_hir_sample.relation)

                    candidate_outside_shuffled_confidences.append(outside_shuffle_confidence)

                    complete_shuffle_hir_sample = hir_sample.complete_shuffle(
                        orig_tokens=sample.tokens,
                        orig_relation=sample.relation,
                        orig_idx=hir_sample.token_idx)

                    complete_shuffle_confidence = self._calculate_confidence(
                        complete_shuffle_hir_sample.tokens,
                        complete_shuffle_hir_sample.relation)

                    candidate_complete_shuffled_confidences.append(complete_shuffle_confidence)

            except ValueError:
                i = 10

        return Explanation(
            sample=sample,
            full_confidence=full_confidence,
            extended_confidence=extended_confidence,
            reduced_confidence=reduced_confidence,
            saliency_interpretations=saliency_interpretations,
            indices_in_reduction=indices_in_reduction,
            candidate_area_confidences=candidate_area_confidences,
            candidate_outside_shuffled_confidences=candidate_outside_shuffled_confidences,
            candidate_complete_shuffled_confidences=candidate_complete_shuffled_confidences,
            hir_samples=hir_samples
        )

    def __hash__(self):
        return hash(f"{self.include_saliency}{self.include_reducer}") ^ \
            hash(self.predictor)


def verify_file_extension(filename, extension):
    if not filename.endswith(extension):
        raise argparse.ArgumentTypeError(f"The file must end with {extension}")
    return filename


def perform_computation(model_path, test_data_path, result_path, subset_size=None):
    # Here is where you implement your actual NLP computation
    print(f"Performing computation with model at {model_path} on test data at {test_data_path}")
    print(f"Results will be saved at {result_path}")

    if subset_size is not None:
        print(f"Only computing on a subset of size {subset_size}")

    predictor_path = Path(model_path)

    predictor = RelationClassifierPredictor.from_path(predictor_path,
                                                      predictor_name="relation_classifier")

    explainer = Explainer(predictor,
                          include_saliency=True,
                          include_reducer=True,
                          include_rule_based_reduction=True
                          )

    data_path = cached_path(test_data_path)

    sample_generator = list(srsly.read_jsonl(data_path))
    if subset_size is not None:
        sample_generator = sample_generator[:subset_size]

    result_samples = []

    for json in tqdm(sample_generator):
        sample = DataSample.from_json(json)

        explanaition = explainer.explain(sample)
        result_samples.append(explanaition.to_json())

    srsly.write_jsonl(result_path, result_samples)


def main():

    import_module_and_submodules("reex")

    parser = argparse.ArgumentParser(description='Perform NLP computation on test data.')
    parser.add_argument('model_path', type=lambda fn: verify_file_extension(fn, '.tar.gz'),
                        help='The location of the NLP model file.')
    parser.add_argument('test_data_path', type=lambda fn: verify_file_extension(fn, '.jsonl'),
                        help='The file path of the test data.')
    parser.add_argument('result_path', type=lambda fn: verify_file_extension(fn, '.jsonl'),
                        help='The path to save results.')
    parser.add_argument('--subset_size', type=int, help='Optional parameter to compute only on subset of test data.',
                        default=None)

    args = parser.parse_args()

    perform_computation(args.model_path, args.test_data_path, args.result_path, args.subset_size)


if __name__ == "__main__":
    main()
