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

        for (int i = 0; i < data.size(); i++) {
            auto &d = data[i];
            if (i < perc * data.size()) {
                classifier_data.push_back(d);
            }
        }

        fit_classifiers(classifier_data, ranges, categorical_features, numerical_features);
    }

    void fit_classifiers(data_t &data,
                         const vector<range_t > &ranges,
                         vector<int> categorical_features = vector<int>(),
                         vector<int> numerical_features = vector<int>()) {
        for (int i = 0; i < classifiers.size(); i++) {
            double factor = (double)i / classifiers.size();
            fit_classifier(&classifiers[i],
                           data,
                           ranges,
                           factor,
                           categorical_features,
                           numerical_features);
        }
    }

    void fit_classifier(RandomFuzzyTree *classifier,
                        data_t data,
                        vector<range_t > ranges,
                        double factor,
                        vector<int> categorical_features = vector<int>(),
                        vector<int> numerical_features = vector<int>()) {
        data_t data_sample = random_sample(data, factor);
        classifier->fit(data_sample, ranges, categorical_features, numerical_features);
    }

    data_t random_sample(data_t &data, double factor = 1) {
        vector<item_t > sample;
        for (int i = 0; i < factor * data.size(); i++) {
            int rand_ind = rand();
            int n = (int) data.size();

            while (rand_ind >= RAND_MAX - (RAND_MAX % n)) {
                rand_ind = rand();
            }
            rand_ind %= n;

            sample.push_back(data[rand_ind]);
        }

        return sample;
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

        if (membs_per_class.size() == 0) {
            auto weights = classifiers[0].root.weights;
            string max_label;
            double max_memb = -10000;

            for (auto &kv : weights) {
                if (kv.second > max_memb) {
                    max_memb = kv.second;
                    max_label = kv.first;
                }
            }

            return max_label;
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
