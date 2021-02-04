import tabula

#dfs = tabula.read_pdf("test/er.pdf", pages='all')
#print(dfs)

#output = "test/test.csv"
tabula.convert_into("test/er.pdf", "test/test.json", output_format="json", pages='all')