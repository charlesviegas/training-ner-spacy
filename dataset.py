import json

import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin

from utils import PathUtil


class SpacyDatasetCreator:
    def __init__(self):
        self.filename = PathUtil.build_path('resources', 'annotations.json')
        self.parser = AnnotationParser()
        self.converter = SpacyConverter()

    def execute(self):
        annotations = self.parser.parse(self.filename)
        train, dev = self.split(annotations)
        docs = self.converter.convert(train)
        self.save(docs, 'train.spacy')
        docs = self.converter.convert(dev)
        self.save(docs, 'dev.spacy')

    @staticmethod
    def split(annotations):
        return train_test_split(annotations, test_size=0.25, random_state=1000, shuffle=True)

    @staticmethod
    def save(docs, filename):
        PathUtil.create_dir(PathUtil.get_root_path(), 'outputs')
        filepath = PathUtil.build_path('outputs', filename)
        docs.to_disk(filepath)


class AnnotationParser:
    @staticmethod
    def parse(filepath):
        file = open(filepath)
        data = json.load(file)
        items = data['annotations']
        file.close()
        return items


class SpacyConverter:
    def __init__(self):
        self.nlp = spacy.blank('pt')

    def convert(self, annotations: list) -> DocBin:
        docs = DocBin()
        for annotation in annotations:
            text = annotation[0]
            entities = annotation[1]['entities']
            doc = self.nlp.make_doc(text)
            doc = self.add_entity_spans(doc, entities)
            docs.add(doc)
        return docs

    @staticmethod
    def add_entity_spans(doc, entities):
        spans = []
        for entity in entities:
            start = entity[0]
            end = entity[1]
            label = entity[2]
            span = doc.char_span(start, end, label=label, alignment_mode='expand')
            if span:
                spans.append(span)
        doc.ents = spans
        return doc


if __name__ == '__main__':
    creator = SpacyDatasetCreator()
    creator.execute()
