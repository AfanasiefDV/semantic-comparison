import json
import copy
def toUniqueForm(str : str) -> str:
    return "".join(
        [''.join([ch for ch in i if ch not in '!"$%&\n\'(),.:;<=>?@[\]^_`{|}~-']) for i in str.lower().split()])

def isInText(str : str, pattern : str) -> bool:
    return not str.find(toUniqueForm(pattern)) == -1


if __name__ == '__main__':
    str = toUniqueForm("Web-технологии")
    # Tests
    assert isInText(str, "web технологии")
    assert isInText(str, "webтехнологии")
    assert isInText(str, "web,технологии")
    assert not isInText(str, "webдехнологии")

    with open('base.json', errors='ignore') as json_file:
        data = json.load(json_file)

    shift = 8
    size = len(data)
    for i in range(len(data)):
        text = toUniqueForm(data[i]["text"])
        for j in range(len(data[i]["keywords"])):
            data[i]["keywords"][j]["isInText"] = isInText(text, data[i]["keywords"][j]["word"])
        index = (i + shift) % size
        for k in range(5):
            obj = copy.copy(data[index]["keywords"][k])
            obj["isInText"] = isInText(text, obj["word"])
            obj["isMatch"] = False
            data[i]["keywords"].append(obj)

    with open('benchmark.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
