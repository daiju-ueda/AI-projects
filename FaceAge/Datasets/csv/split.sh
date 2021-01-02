#!/bin/zsh
all_no=`cat random.csv | wc -l`
train_no=$((all_no * 0.8))
integer val_no=$((all_no * 0.1))
integer test_no=$((all_no * 0.1))
train_no=$((all_no - val_no - test_no))

cat random.csv | head -n $train_no | gsed -e "1i id,label"> complete_regression_train.csv
cat random.csv | tail -n $test_no | gsed -e "1i id,label"> complete_regression_test.csv
cat random.csv | tail -n $((all_no - train_no)) | head -n $val_no | gsed -e "1i id,label"> complete_regression_val.csv
