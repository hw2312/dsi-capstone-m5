import src.preprocessing as prep
import src.constants as constants


def extract_sales():
    print('IS_PROTOTYPING', constants.IS_PROTOTYPING)
    prep.extract_sales()
    prep.extract_sales_pred()
