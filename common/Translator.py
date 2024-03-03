from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class Translator:
    def __init__(self):
        self.source = 'eng_Latn'
        self.target = 'rus_Cyrl'
        self.model_name = 'facebook/nllb-200-distilled-600M'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.translator = pipeline(
            'translation', model=self.model, tokenizer=self.tokenizer, src_lang=self.source, tgt_lang=self.target)

    def translate(self, doc):
        output = self.translator(doc, max_length=512)
        return ' '.join(output[i]['translation_text'] for i in range(len(output)))

