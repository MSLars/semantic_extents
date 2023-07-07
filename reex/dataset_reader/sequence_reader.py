from typing import Dict, List, Union, Iterable

import srsly
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField, LabelField, TransformerTextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from cached_path import cached_path
from transformers import AutoTokenizer


@DatasetReader.register("sequence", exist_ok=True)
class PreParsedReader(DatasetReader):
    """
    This class reads pre-parsed sentences
    """

    def __init__(
            self,
            max_tokens: int = None,
            token_namespace: str = "tokens",
            label_namespace: str = "labels",
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        self.max_tokens = max_tokens

        self.token_namespace = token_namespace
        self.label_namespace = label_namespace

    def text_to_instance(
            self,
            text: List[str],
            relation: List[Union[str, int]],
    ) -> Instance:

        if isinstance(text, str):
            raise ValueError("text_to_instance expects a list of tokens. The indices in the relation are based on this "
                             "Tokenization.")
        else:
            tokens = [t for t in text]

        if relation is None:
            raise ValueError("This model classifies relations. This implies that the relation attribute must at least"
                             "contain the indices in the form (start_arg1, end_arg1, start_arg2, end_arg2).")

        relation_indices = sorted([int(i) for i in relation[:4]])

        parts = [tokens[:relation_indices[0]],
                 tokens[relation_indices[0]:relation_indices[1] + 1],
                 tokens[relation_indices[1] + 1:relation_indices[2]],
                 tokens[relation_indices[2]:relation_indices[3] + 1],
                 tokens[relation_indices[3] + 1:]]

        word_offset = 0
        model_tokens = parts[0] + ["</s>"]
        part_encoding = self.tokenizer(parts[0], is_split_into_words=True, return_offsets_mapping=True)

        input_ids = part_encoding["input_ids"]
        transformed_relation_indices = []
        wordpiece_relation_indices = []
        token_type_ids = part_encoding.encodings[0].type_ids
        attention_mask = part_encoding["attention_mask"]
        special_tokens_mask = part_encoding.encodings[0].special_tokens_mask
        offset_mapping = part_encoding["offset_mapping"]
        word_ids = [i + word_offset if i is not None else None for i in part_encoding.word_ids()]
        if word_ids[-1] is None:
            if word_ids[-2] is None:
                word_ids[-1] = 0
            else:
                word_ids[-1] = word_ids[-2] + 1

        word_offset = word_ids[-1] + 1

        # word offset läuft noch nicht
        for seq_idx, part in enumerate(parts[1:]):

            part_encoding = self.tokenizer(part, is_split_into_words=True, return_offsets_mapping=True)

            if seq_idx in [0, 2]:
                transformed_relation_indices += [len(input_ids), len(input_ids) + len(part_encoding["input_ids"])-3]
                wordpiece_relation_indices += [len(model_tokens), len(model_tokens) + len(part) - 1]

            model_tokens = model_tokens + part + ["</s>"]
            input_ids.extend(part_encoding["input_ids"][1:])

            token_type_ids.extend(part_encoding.encodings[0].type_ids[1:])
            attention_mask.extend(part_encoding["attention_mask"][1:])
            special_tokens_mask.extend(part_encoding.encodings[0].special_tokens_mask[1:])
            offset_mapping.extend(part_encoding["offset_mapping"][1:])
            word_ids.extend([i + word_offset if i is not None else None for i in part_encoding.word_ids()[1:]])
            if word_ids[-1] is None:
                word_ids[-1] = word_ids[-2] + 1
            word_offset = word_ids[-1] + 1

        text_field = TransformerTextField(input_ids,
                                          token_type_ids,
                                          attention_mask,
                                          # special_tokens_mask,
                                          # offset_mapping,
                                          )

        metadata = MetadataField({"model_tokens": model_tokens,
                                  "word_ids": word_ids,
                                  "word_pieces": self.tokenizer.convert_ids_to_tokens(input_ids),
                                  "transformed_relation_indices": transformed_relation_indices,
                                  "wordpiece_relation_indices": wordpiece_relation_indices,
                                  "indices": [i for i, _ in enumerate(input_ids)]})

        if len(relation) == 5:
            label_field = LabelField(relation[4], label_namespace=self.label_namespace)

            instance = Instance({"tokens": text_field, "label": label_field, "metadata": metadata})

        else:
            instance = Instance({"tokens": text_field, "metadata": metadata})

        return instance

    def _read(self, file_path: str) -> Iterable[Instance]:

        cp = cached_path(file_path)

        iterable_corpus = srsly.read_jsonl(cp)

        for sample in iterable_corpus:  # pylint: disable=unused-variable

            yield self.text_to_instance(
                text=sample["tokens"],
                relation=sample["relation"],
            )


if __name__ == "__main__":
    train_url = "https://fh-aachen.sciebo.de/s/YHTZEF5ahUtEO70/download"

    reader = PreParsedReader()

    train_dataset = list(reader.read(train_url))

    i = 10
