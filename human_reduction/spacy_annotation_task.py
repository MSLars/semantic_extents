import itertools
from pathlib import Path
import random
from typing import Union, Iterator, List

import spacy
import srsly
from allennlp.common.file_utils import cached_path
from allennlp.common.util import import_module_and_submodules
from spacy.tokens import Doc, Token, Span

import pretrained_models
from explain.explanaition import DataSample, Explainer
from human_reduction.annotation_task_datamodel import HIRAnnotationSample, HIRLabel, \
    HIRAnnotationTask
from reex.predictor.sequence_predictor import RelationClassifierPredictor


nlp = spacy.load("en_core_web_sm")

MAX_ITERATIONS = 30


def sample_from_tokens(sample_tokens, relation_label, description=""):

    def get_token_index(token):
        if isinstance(token, Token):
            return token.i
        elif isinstance(token, Span):
            return token.start

    sorted_samples = sorted(sample_tokens, key=get_token_index)

    res_tokens = []
    res_idx = []
    relation_idx = []

    index = 0

    for elem in sorted_samples:
        if isinstance(elem, Token):
            elem: Token
            res_tokens.append(elem.text)
            res_idx.append(elem.i)
            index += 1

        elif isinstance(elem, Span):
            relation_idx.append(index)
            for token in elem:
                token: Token
                res_tokens.append(token.text)
                res_idx.append(token.i)
                index += 1
            relation_idx.append(index - 1)

    return HIRAnnotationSample(tokens=res_tokens,
                               relation=relation_idx + [relation_label],
                               description=description,
                               token_idx=res_idx)


def extend_sample_tokens(sample_tokens, token):
    all_tokens = []
    for elem in sample_tokens:
        if isinstance(elem, Token):
            all_tokens.append(elem)
        elif isinstance(elem, Span):
            all_tokens.extend([t for t in elem])

    if not token in all_tokens:
        sample_tokens.append(token)
        return True

    return False

verb_pos = ["AUX", "VERB"]


def create_sample_iterator(sample: DataSample) -> Iterator[HIRAnnotationSample]:

    all_samples = []

    doc = Doc(nlp.vocab, words=sample.tokens)
    doc = nlp(doc)

    inclusive_relation_indices_sorted = sorted(sample.relation[:4])

    h_start, h_end = (inclusive_relation_indices_sorted[0], inclusive_relation_indices_sorted[1] + 1)
    t_start, t_end = (inclusive_relation_indices_sorted[2], inclusive_relation_indices_sorted[3] + 1)

    head = doc[h_start:h_end]
    tail = doc[t_start:t_end]

    sample_tokens = [head, tail]
    all_samples.append(sample_from_tokens(sample_tokens, sample.relation[4], "ONLY_ARGS"))

    head_subtree = list(head.root.subtree)
    tail_subtree = list(tail.root.subtree)

    for subtree in [head_subtree, tail_subtree]:
        for token in subtree:
            sample_extended = extend_sample_tokens(sample_tokens, token)
            if sample_extended:
                all_samples.append(
                    sample_from_tokens(sample_tokens, sample.relation[4], "ARGS_SUBTREE"))

    import networkx
    edges = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT":
                target = token
            for child in token.children:
                edges.append((token, child))

    for e in list(itertools.combinations([s.root for s in doc.sents], 2)):
        edges.append(e)

    graph = networkx.Graph(edges)
    try:
        sp = networkx.shortest_path(graph, source=head.root, target=tail.root)
    except networkx.exception.NetworkXNoPath:
        i = 10

    verbs_in_path = [token for token in sp if token.pos_ in verb_pos]
    for token in verbs_in_path:
        sample_extended = extend_sample_tokens(sample_tokens, token)
        if sample_extended:
            all_samples.append(
                sample_from_tokens(sample_tokens, sample.relation[4], "VERB_ON_PATH"))

    for token in sp:
        sample_extended = extend_sample_tokens(sample_tokens, token)
        if sample_extended:
            all_samples.append(
                sample_from_tokens(sample_tokens, sample.relation[4], "ARG_PATH"))

    _, start, end, _ = sorted([h_start, h_end, t_start, t_end])
    for token in doc[start:end]:
        sample_extended = extend_sample_tokens(sample_tokens, token)
        if sample_extended:
            all_samples.append(
                sample_from_tokens(sample_tokens, sample.relation[4], "BETWEEN_ARGS"))

    for token in doc[sample.extend[0]:sample.extend[1]+1]:
        sample_extended = extend_sample_tokens(sample_tokens, token)
        if sample_extended:
            all_samples.append(
                sample_from_tokens(sample_tokens, sample.relation[4], "EXTEND"))

    for token in doc:

        distances_to_args = [abs(token.i - h_start),
                             abs(token.i - h_end),
                             abs(token.i - t_start),
                             abs(token.i - t_end)]

        if min(distances_to_args) > 5:
            continue

        sample_extended = extend_sample_tokens(sample_tokens, token)
        if sample_extended:
            all_samples.append(
                sample_from_tokens(sample_tokens, sample.relation[4], "ALL"))

    result_samples = all_samples[:MAX_ITERATIONS]

    all_description_categories = ["ONLY_ARGS", "ARGS_SUBTREE", "VERB_ON_PATH", "ARG_PATH", "BETWEEN_ARGS", "EXTEND", "ALL"]

    all_description_category_counts = {category:
                                           len([s for s in result_samples
                                                if s.description == category])
                                       for category in all_description_categories}

    all_description_category_counter = {category: 0 for category in all_description_categories}

    for sample in result_samples:
        all_description_category_counter[sample.description] += 1
        tmp_count = all_description_category_counter[sample.description]
        tmp_n = all_description_category_counts[sample.description]
        sample.count_description = f"{sample.description} ({tmp_count}/{tmp_n})"

    return all_samples[:30]


if __name__ == '__main__':

    import_module_and_submodules("reex")

    test_data_path: str = "https://fh-aachen.sciebo.de/s/Z1Q2kkerjM4zPPd/download"
    test_data_path = cached_path(test_data_path)

    predictor_path = Path(pretrained_models.__file__).parent / "reex_roberta_v1.tar.gz"

    predictor = RelationClassifierPredictor.from_path(predictor_path, predictor_name="relation_classifier")

    explainer = Explainer(predictor, False, False)

    tasks = []

    complete_corpus = list(srsly.read_jsonl(test_data_path))

    all_relation_labels = {s["relation"][4] for s in complete_corpus}

    long_extend_samples = [c for c in complete_corpus
                           if c["lexicalcondition"] in ["Verbal", "Other"]]

    short_extend_samples = [c for c in complete_corpus
                           if c["lexicalcondition"] not in ["Verbal", "Other"]]

    experiment_samples = []
    n_long = 0
    n_short = 0

    for label in all_relation_labels:

        le = [s for s in long_extend_samples if s["relation"][4] == label]
        experiment_samples.extend(random.sample(le, min(len(le), 6)))
        n_long += min(len(le), 6)

        se = [s for s in short_extend_samples if s["relation"][4] == label]
        experiment_samples.extend(random.sample(se, min(len(se), 6)))
        n_short += min(len(se), 6)

    experiment_samples.extend(random.sample(
        [l for l in long_extend_samples if l not in experiment_samples],
        110 - n_long))

    experiment_samples.extend(random.sample(
        [l for l in short_extend_samples if l not in experiment_samples],
        110 - n_short))

    NUM_CHOICES = 3

    random.shuffle(experiment_samples)

    for i, sample_dict in enumerate(experiment_samples):
        sample = DataSample(**sample_dict)

        # if sample.lexicalcondition != "Verbal":
        #     continue

        orig_label = sample.relation[4]

        try:
            hir_samples = create_sample_iterator(sample)
        except ValueError:
            i = 10
            continue
        explaination = explainer.explain(sample)

        sorted_predictions = sorted(
            [(label, prob) for (label, prob) in explaination.full_confidence.probabilities.items()],
            key=lambda x: x[1], reverse=True)

        if sample.relation[4] not in [l[0] for l in sorted_predictions[:3]]:

            gold_prob = explaination.full_confidence.probabilities[sample.relation[4]]
            gold_label = sample.relation[4]

            sorted_predictions[2] = (gold_label, gold_prob)

        hir_choices = [HIRLabel(label=label, prob=prob) for (label, prob) in sorted_predictions[:3]]

        assert sample.relation[4] in [h.label for h in hir_choices]

        task = HIRAnnotationTask(hir_iterator=hir_samples,
                                 orig_sample=sample,
                                 choices=hir_choices,
                                 id=sample.id,)

        tasks += [task]

    output_path = Path(__file__).parent / "hir_annotation_spacy.jsonl"

    task_dicts = [task.to_json() for task in tasks]

    srsly.write_jsonl(output_path, task_dicts)
