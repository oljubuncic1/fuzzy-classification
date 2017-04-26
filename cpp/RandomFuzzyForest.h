#ifndef CPP_RANDOMFUZZYFOREST_H
#define CPP_RANDOMFUZZYFOREST_H

#include "includes.h"
#include "RandomFuzzyTree.h"
#include "definitions.h"

class RandomFuzzyForest {
private:
    vector<RandomFuzzyTree> classifiers;
    int job_n;
public:
    RandomFuzzyForest(int classifier_n, int job_n = 4) {
        classifiers = vector<RandomFuzzyTree>(classifier_n);
        this->job_n = job_n;
    }

    void fit(data_t &data,
             vector<range_t > &ranges,
             vector<int> categorical_features = vector<int>(),
             vector<int> numerical_features = vector<int>()) {
        double perc = 1.0;

        data_t classifier_data;
        data_t weight_data;

        for (int i = 0; i < data.size(); i++) {
            auto &d = data[i];
            if (i < perc * data.size()) {
                classifier_data.push_back(d);
            } else {
                weight_data.push_back(d);
            }
        }

        fit_classifiers(classifier_data, ranges, categorical_features, numerical_features);
        fit_weights(weight_data);
    }

    void fit_classifiers(data_t &data,
                         const vector<range_t > &ranges,
                         vector<int> categorical_features = vector<int>(),
                         vector<int> numerical_features = vector<int>()) {
        bool parallel = false;
        if(parallel) {
            vector<thread> threads(classifiers.size());

            for (int i = 0; i < ceil((double) classifiers.size() / job_n); i++) {
                for (int j = 0; j < job_n; j++) {
                    int curr_ind = i * job_n + j;
                    if (curr_ind < classifiers.size()) {
                        threads[j] = thread([this, data, ranges, curr_ind, categorical_features, numerical_features] {
                            fit_classifier(&classifiers[curr_ind],
                                           data,
                                           ranges,
                                           categorical_features,
                                           numerical_features);
                        });
                    }
                }

                for (int j = 0; j < job_n; j++) {
                    int curr_ind = i * job_n + j;
                    if (curr_ind < classifiers.size()) {
                        threads[j].join();
                    }
                }
            }
        } else {
            for(int i = 0; i < classifiers.size(); i++) {
                fit_classifier(&classifiers[i],
                              data,
                              ranges,
                              categorical_features,
                              numerical_features);

                int dmy = 1;
            }
        }
    }

    void fit_weights(data_t &data) {
        double alpha = 0;

        for (auto &d : data) {
            string y = d.second;

            for (auto &classifier : classifiers) {
                map<string, double> prediction = classifier.predict_memberships(d);

                double total = 0;
                auto &weights = classifier.weights();
                for (auto &kv : prediction) {
                    total += weights[kv.first] * kv.second;
                }

                for (auto kv : weights) {
                    double t;
                    if (kv.first.compare(y) == 0) {
                        t = total;
                    } else {
                        t = 0;
                    }

                    double dy = weights[kv.first] * prediction[kv.first] - t;
                    double delta_w = -alpha * dy * prediction[kv.first];

                    weights[kv.first] += delta_w;
                }
            }
        }
    }

    void fit_classifier(RandomFuzzyTree *classifier,
                        data_t data,
                        vector<range_t > ranges,
                        vector<int> categorical_features = vector<int>(),
                        vector<int> numerical_features = vector<int>()) {
        data_t data_sample = random_sample(data);
        classifier->fit(data_sample, ranges, categorical_features, numerical_features);
    }

    data_t random_sample(data_t &data) {
        set<item_t > sample;
        for (int i = 0; i < data.size(); i++) {
            int rand_ind = rand();
            int n = (int) data.size();

            while (rand_ind >= RAND_MAX - (RAND_MAX % n)) {
                rand_ind = rand();
            }
            rand_ind %= n;

            sample.insert(data[rand_ind]);
        }

        data_t sample_data;
        copy(sample.begin(), sample.end(), back_inserter(sample_data));

        return sample_data;
    }

    string predict(item_t &x) {
        map<string, double> membs_per_class;

        for (int i = 0; i < classifiers.size(); i++) {
            auto classifier = classifiers[i];
            map<string, double> membs = classifier.predict_memberships(x);
            for (auto kv : membs) {
                string cls = kv.first;
                double val = kv.second;

                membs_per_class[cls] += val;
            }
        }

        return std::max_element(membs_per_class.begin(),
                                membs_per_class.end(),
                                [](const pair<string, int> &p1, const pair<string, int> &p2) {
                                    return p1.second < p2.second;
                                })->first;
    }

    double score(data_t &data) {
        int correct_n = 0;
        for (item_t &d : data) {
            string prediction = predict(d);
            if (prediction.compare(d.second) == 0) {
                correct_n++;
            }
        }

        return (double) correct_n / data.size();
    }

};

#endif //CPP_RANDOMFUZZYFOREST_H
