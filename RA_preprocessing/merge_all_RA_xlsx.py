import numpy as np
import pandas as pd
import glob

xlsx_files = []
for xlsx_file in glob.glob('RA_Jun2/**/*.xlsx'):
	xlsx_files.append(pd.read_excel(xlsx_file))

all_xlsx_files = pd.concat(xlsx_files)
all_xlsx_files.to_excel('all_CATCH.xlsx')
