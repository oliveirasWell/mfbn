rm -rf output
zip models.zip ./models/*
spark-submit --py-files models.zip  mfbn.py -cnf ./input/moreno-1.json