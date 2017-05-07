a=`hadoop fs -cat s3://farukintermediate/step-3-output/* | wc -l`
b=`hadoop fs -cat s3://farukintermediate/step-1-output/* | grep -o 't' | wc -l`

echo 'print $a.0 / $b' | python