import json
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

    with open('benchmark.json', errors='ignore') as json_file:
        data = json.load(json_file)

    for i in range(len(data)):
        text = toUniqueForm(data[i]["text"])
        for j in range(len(data[i]["keywords"])):
            data[i]["keywords"][j]["isInText"] = isInText(text, data[i]["keywords"][j]["word"])
    print(data)
    size = len(data)
    shift = 8
    for i in range(size):
        index = (i + shift) % size
        obj = data[i]
        obj["keywords"] = data[index]["keywords"]
        obj["id"] = i+size
        data.append(obj)
        print(len(data))
    with open('benchmark.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
