from common.TextPreproccessing import TextPreprocessing
import json

def lematized():
    textPreprocessing = TextPreprocessing()
    lemaDataset = []
    with open('doc.json', errors='ignore', encoding='utf-8') as file:
        data = json.load(file)
         
    for i in range(len(data)):
        lemaDataset.append(textPreprocessing.preprocess(data[i]["text"]).split())

    with open('docLema.json', 'w', encoding='utf-8') as file:
        json.dump(lemaDataset, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    lematized()

