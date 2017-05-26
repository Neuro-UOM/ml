from importer import csvImporter

dataImporter = csvImporter('train.csv')
data = dataImporter.getData()
print(data[0])

input("Press Enter to continue...")