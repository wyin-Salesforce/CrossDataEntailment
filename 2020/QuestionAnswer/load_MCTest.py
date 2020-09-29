import xml.etree.ElementTree as ET

def load_MCTest():
    path  = '/export/home/Dataset/MCTest/Statements/'
    tree = ET.parse(path+'mc500.train.statements.pairs')
    root = tree.getroot()
    for child in root:
        print(child)
        exit(0)

if __name__ == "__main__":
    load_MCTest()
