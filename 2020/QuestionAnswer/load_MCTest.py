import xml.etree.ElementTree as ET

def load_MCTest():
    path  = '/export/home/Dataset/MCTest/Statements/'
    tree = ET.parse(path+'mc500.train.statements.pairs')
    root = tree.getroot()
    for pair in root.findall('pair'):
        premise = pair.find('t').text
        hypothesis = pair.find('h').text
        label = pair.entailment
        print(premise)
        print(hypothesis)
        print('label:', label)
        exit(0)

if __name__ == "__main__":
    load_MCTest()
