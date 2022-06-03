import preprocess
def return_ans(name,company,year,kms_driven,fuel_type):
# name='Maruti Suzuki Swift'
# company="Maruti"
# year=2019
# kms_driven=100
# fuel_type="Petrol"
    ans=preprocess.find_ans(name,company,year,kms_driven,fuel_type)
    # print("yes this is the value:",ans)
    return ans