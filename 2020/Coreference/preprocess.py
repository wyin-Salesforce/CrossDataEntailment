import csv

def load_GAP_coreference_data():
    path = '/export/home/Dataset/gap_coreference/'

    with open(path+'gap-development.tsv') as tsvfile:
      reader = csv.DictReader(tsvfile, dialect='excel-tab')
      for row in reader:
          print(row)


if __name__ == "__main__":
    load_GAP_coreference_data()
