$1 = ${hadoop fs -cat s3://farukintermediate/step-3-output/* | wc -l}
$2 = ${hadoop fs -cat s3://farukintermediate/step-1-output/* | grep -o 't' | wc -l}

echo "print 1.0 * $1 / $2" | python