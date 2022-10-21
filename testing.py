import spacy

from utils import PathUtil
filepath = PathUtil.build_path('outputs', 'model-best')

nlp = spacy.load(filepath)

text = (
    'BACON FATIADO MEDALHAO 1KG'
    '(40)COXINHA DA ASA CONG. IN NAT. INTERF. CX 18 KG BELLO'
    '(346) MEIO DA ASA CONG. TEMP.PCT 750 G CX 18 KG'
    '(229) COXA C/SOBRECOXA INDIVIDUAL CONGELADO CX 18 KGS BELLO'
    '(18)CORTES CONG. DE FRANGO BDJ - MEIO DAS ASAS CX 12 KG'
    '1617 - CORTES CONG.FRANGO COXA/SOB S/OSSO BDJ BELLO CX 9,6KG'
)

doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)
