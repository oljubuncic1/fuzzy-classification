g++ --std=c++11 data-distributor/mapper.cpp -o /usr/local/bin/measurementsbin/step-1-mapper
g++ --std=c++11 mapper/mapper.cpp -o /usr/local/bin/measurementsbin/step-2-mapper
g++ --std=c++11 reducer/main.cpp reducer/*.h -o /usr/local/bin/measurementsbin/step-2-reducer
g++ --std=c++11 result_reducer/reducer.cpp -o /usr/local/bin/measurementsbin/step-3-reducer

hadoop fs -rm -R -f s3://farukbin/bin/*
hadoop fs -rm -R -f s3://farukintermediate/*
hadoop fs -put /usr/local/bin/measurementsbin/* s3://farukbin/bin
