import random
from dataclasses import dataclass
from typing import Union, List

from pydantic import BaseModel


@dataclass
class DataSample:
    tokens: list[str]
    relation: list[Union[str, int]]
    extend: list[int]
    lexicalcondition: str
    arg1_info: str = None
    arg2_info: str = None
    id: str = None

    def __hash__(self):
        return hash(tuple(self.tokens)) ^ \
            hash(tuple(self.relation)) ^ \
            hash(tuple(self.extend)) ^ \
            hash(self.lexicalcondition) ^ \
            hash(self.arg1_info) ^ \
            hash(self.arg2_info) ^ \
            hash(self.id)

    def to_json(self):
        return {
            "tokens": self.tokens,
            "relation": self.relation,
            "extend": self.extend,
            "lexicalcondition": self.lexicalcondition,
            "arg1_info": self.arg1_info,
            "arg2_info": self.arg2_info,
            "id": self.id
        }

    @staticmethod
    def from_json(json_elem):
        return DataSample(tokens=json_elem["tokens"],
                          relation=json_elem["relation"],
                          extend=json_elem["extend"],
                          lexicalcondition=json_elem["lexicalcondition"],
                          arg1_info=json_elem["arg1_info"],
                          arg2_info=json_elem["arg2_info"],
                          id=json_elem["id"])

    @property
    def label(self):
        return self.relation[4]


class HIRLabel(BaseModel):
    label: str
    prob: float
    explanation: str = "Hier könnte Ihre Werbung stehen."

    def to_json(self):
        return self.dict()

    @staticmethod
    def from_json(json):
        return HIRLabel(label=json["label"], prob=json["prob"], explanation=json["explanation"])


class HIRAnnotationSample(BaseModel):
    tokens: list[str]
    relation: list[Union[str, int]]
    description: str = ""
    count_description: str = ""
    token_idx: List[int] = None

    def to_json(self):
        return {
            "tokens": self.tokens,
            "relation": self.relation,
            "description": self.description,
            "count_description": self.count_description,
            "token_idx": self.token_idx if self.token_idx else []
        }

    @staticmethod
    def from_json(json_elem):

        if "tokens" not in json_elem:
            raise ValueError("Invalid json. Missing key 'tokens'")

        return HIRAnnotationSample(tokens=json_elem["tokens"],
                                   relation=[
                                       int(json_elem["relation"][0]),
                                       int(json_elem["relation"][1]),
                                       int(json_elem["relation"][2]),
                                       int(json_elem["relation"][3]),
                                       json_elem["relation"][4]
                                   ],
                                   description=json_elem["description"],
                                   count_description=json_elem["count_description"],
                                   token_idx=json_elem["token_idx"] if "token_idx" in json_elem else None)

    def outside_shuffle(self, orig_tokens, orig_relation):

        prior_insertions = orig_relation[0] - int(self.relation[0])

        tokens_to_insert = [t for t in orig_tokens]

        prior_tokens = random.sample(tokens_to_insert, prior_insertions)

        posterior_insertions = len(orig_tokens) - orig_relation[3]
        posterior_tokens = random.sample(tokens_to_insert, posterior_insertions)

        new_tokens = prior_tokens + self.tokens + posterior_tokens

        new_relation = [
                int(self.relation[0]) + prior_insertions,
                int(self.relation[1]) + prior_insertions,
                int(self.relation[2]) + prior_insertions,
                int(self.relation[3]) + prior_insertions,
                self.relation[4]
            ]

        if " ".join(new_tokens[new_relation[0]:new_relation[3]+1]) != \
               " ".join(self.tokens[int(self.relation[0]):int(self.relation[3])+1]):
                i = 10

        return HIRAnnotationSample(
            tokens=new_tokens,
            relation=[
                int(self.relation[0]) + prior_insertions,
                int(self.relation[1]) + prior_insertions,
                int(self.relation[2]) + prior_insertions,
                int(self.relation[3]) + prior_insertions,
                self.relation[4]
            ],
            description=f"OUTSIDE_SHUFFLE_{self.description}",
            count_description=f"OUTSIDE_SHUFFLE_{self.count_description}"
        )

    def complete_shuffle(self, orig_tokens, orig_relation, orig_idx):
        tokens_to_insert = [t for t in orig_tokens if t not in self.tokens]
        new_tokens = []

        sorted_orig_idx = iter(sorted(orig_idx))
        next_da_tokens = next(sorted_orig_idx)

        for i in range(len(orig_tokens)):

            if i == next_da_tokens:
                new_tokens.append(orig_tokens[next_da_tokens])
                try:
                    next_da_tokens = next(sorted_orig_idx)
                except StopIteration:
                    # this happens for the last element, we dont really care about that
                    pass
            else:
                # TODO here list index out of range
                if not tokens_to_insert:
                    new_tokens.append(random.choice(orig_tokens))
                else:
                    try:
                        new_tokens.append(random.choice(tokens_to_insert))
                    except IndexError:
                        new_tokens.append(random.choice(orig_tokens))

        orig_indices = sorted(orig_relation[:4])

        if " ".join(new_tokens[orig_indices[0]:orig_indices[1]+1]) != \
               " ".join(self.tokens[int(self.relation[0]):int(self.relation[1])+1]):

            i = 10

        if " ".join(new_tokens[orig_indices[2]:orig_indices[3]+1]) != \
               " ".join(self.tokens[int(self.relation[2]):int(self.relation[3])+1]):

            i = 10

        return HIRAnnotationSample(
            tokens=new_tokens,
            relation=orig_relation,
            description=f"COMPLETE_SHUFFLE_{self.description}",
            count_description=f"COMPLETE_SHUFFLE_{self.count_description}"
        )


class HIRAnnotationTask(BaseModel):
    hir_iterator: List[HIRAnnotationSample]
    orig_sample: DataSample

    choices: List[HIRLabel]
    choice: HIRLabel = None
    choice_index: int = 0

    num_extensions: int = 0
    extended_tokens: List[str] = []

    approoved: bool = False

    visual_choices: List[str] = None

    id: str = ""

    def __init__(self,
                 hir_iterator: List[HIRAnnotationSample],
                 orig_sample: DataSample,
                 choices: List[HIRLabel],
                 choice: HIRLabel = None,
                 choice_index: int = 0,
                 num_extensions: int = 0,
                 extended_tokens: List[str] = None,
                 approoved: bool = False,
                 visual_choices: List[str] = None,
                 id: str = None,):

        if extended_tokens is None:
            extended_tokens = []

        super().__init__(hir_iterator=hir_iterator,
                         orig_sample=orig_sample,
                         choices=choices,
                         choice=choice,
                         choice_index=choice_index,
                         num_extensions=num_extensions,
                         extended_tokens=extended_tokens,
                         approoved=approoved,
                         visual_choices=visual_choices,
                         id=id)

        if visual_choices is None:
            visual_choices = [l.label for l in choices]
            random.shuffle(visual_choices)

            visual_choices = ["UNSURE"] + visual_choices

            visual_choices = [f"({i + 1}): {c}" for i, c in enumerate(visual_choices)]

            self.visual_choices = visual_choices

    def to_json(self):
        return {
            "hir_iterator": [hir.to_json() for hir in self.hir_iterator],
            "orig_sample": self.orig_sample.to_json(),
            "choices": [choice.to_json() for choice in self.choices],
            "choice": self.choice.to_json() if self.choice else None,
            "choice_index": self.choice_index,
            "num_extensions": self.num_extensions,
            "extended_tokens": self.extended_tokens,
            "approoved": self.approoved,
            "visual_choices": self.visual_choices,
            "id": self.id
        }

    @staticmethod
    def from_json(json):
        return HIRAnnotationTask(
            hir_iterator=[HIRAnnotationSample.from_json(hir) for hir in json["hir_iterator"]],
            orig_sample=DataSample.from_json(json["orig_sample"]),
            choices=[HIRLabel.from_json(choice) for choice in json["choices"]],
            choice=HIRLabel.from_json(json["choice"]) if json["choice"] else None,
            choice_index=json["choice_index"],
            num_extensions=json["num_extensions"],
            extended_tokens=json["extended_tokens"],
            approoved=json["approoved"],
            visual_choices=json["visual_choices"],
            id=json["id"]
        )
