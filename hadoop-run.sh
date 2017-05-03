$HADOOP_PREFIX/bin/hadoop fs -rm -R -f output* resultoutput* data000*

/opt/rh/devtoolset-2/root/usr/bin/c++ --std=c++11 mapper/mapper.cpp -o /usr/local/bin/mapper && /opt/rh/devtoolset-2/root/usr/bin/c++ --std=c++11 reducer/main.cpp reducer/*.h -o /usr/local/bin/reducer && /opt/rh/devtoolset-2/root/usr/bin/c++ --std=c++11 data-distributor/mapper.cpp -o /usr/local/bin/data_distributor_mapper && /opt/rh/devtoolset-2/root/usr/bin/c++ --std=c++11 result_reducer/reducer.cpp -o /usr/local/bin/result_reducer

$HADOOP_PREFIX/bin/hadoop jar /usr/local/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar -D mapreduce.job.reduces=$2 -cmdenv REDUCERN=$2 -mapper /usr/local/bin/data_distributor_mapper -reducer /bin/cat -input $1 -output data000
$HADOOP_PREFIX/bin/hadoop jar /usr/local/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar -D mapreduce.job.reduces=$2  -cmdenv REDUCERN=$2 -mapper /usr/local/bin/mapper -reducer /usr/local/bin/reducer -input data000 -output output000
$HADOOP_PREFIX/bin/hadoop jar /usr/local/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar  -D mapreduce.job.reduces=$2 -cmdenv REDUCERN=$2 -mapper /bin/cat -reducer /usr/local/bin/result_reducer -input output000 -output resultoutput000

$HADOOP_PREFIX/bin/hadoop fs -cat resultoutput000/part-00000