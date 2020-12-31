#
rm -rf output
rm -f models.zip
rm out.log
zip -r models.zip ./models/* ./input/*
#spark-submit \
#--master local[$1] \
#--py-files models.zip  mfbn.py -cnf ./input/moreno-1-gbm.json \
