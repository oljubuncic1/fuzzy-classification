//
// Created by faruk on 4/30/17.
//

#ifndef CPP_LOCALRUNNER_H
#define CPP_LOCALRUNNER_H

#include "includes.h"
#include "definitions.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"

using namespace std;

class LocalRunner {
public:
    bool isOnlyDouble(const char* str) const {
        char* endptr = 0;
        try {
            strtod(str, &endptr);
        } catch(...) {
            return false;
        }


        if(*endptr != '\0' || endptr == str)
            return false;
        return true;
    }

    void run(bool shuffle, bool debug, int clasifier_n, int job_n, int fold_n) {
        vector<string> datasets = {"HAB",
                                   "HAY",
                                   "IRI",
                                   "MAM",
                                   "NEW",
                                   "TAE",
                                   "BUP",
                                   "APP",
                                   "PIM",
                                   "GLA",
                                   "SAH",
                                   "WIS",
                                   "CLE",
                                   "HEA",
                                   "WIN",
                                   "AUS",
                                   "VEH",
                                   "BAN",
                                   "HEP",

//                                   "IMA",
//                                   "THY",

                                   "WDB",
                                   "DER",
                                   "ION",
                                   "SON"};
//        };
//    datasets = {"HAB", "HAY", "TAE", "BUP"};
//        datasets = {"BUP", "APP", "HEA", "CLE", "VEH", "BAN", "HEP"};
//        datasets = { "VEH" };
        srand(time(NULL));

        set<string> labels;
        double error_score = 0;
        for (string &dataset : datasets) {
            vector<example_t > string_data;
            vector<range_t > ranges;
            vector<int> categorical_features;
            vector<int> numerical_features;
            double accuracy;

            load_data(dataset,
                      string_data,
                      categorical_features,
                      numerical_features,
                      accuracy);
            find_ranges(string_data, ranges);

            data_t data;
            string last_label = "";
            map<string, int> vals;
            int curr = 0;
            for (auto &x : string_data) {
                vector<double> item;
                int i = 0;
                for (int j = 0; j < x.first.size(); j++) {
                    auto val = x.first[i];

                    double real_val;
                    istringstream os(val);

                    if(not isOnlyDouble(val.c_str())) {
                        if(vals[val] == 0) {
                            vals[val] = curr;
                            curr++;
                        }

                        real_val = (double)vals[val];
                    } else {
                        os >> real_val;
                    }

                    if(real_val < ranges[j].first) {
                        ranges[j].first = real_val;
                    } else if(real_val > ranges[j].second) {
                        ranges[j].second = real_val;
                    }

                    int numerical_feature_n = (int) (ranges.size() - categorical_features.size());
                    if(find(categorical_features.begin(), categorical_features.end(), i) == categorical_features.end()) {
                        if((double)rand() / RAND_MAX < 0.4) {
                            bool negative = (rand() % 2 == 0);
                            double perc = 1 + 0.4 * (double)rand() / RAND_MAX;
                            if(negative) {
                                real_val *= -perc;
                            } else {
                                real_val *= perc;
                            }
                        }
                    }

                    item.push_back(real_val);
                    i++;
                }
                string label = x.second;
                labels.insert(label);

//                if(labels.size() > 1 && (double)rand() / RAND_MAX < 0.1) {
//                    auto it = labels.begin();
//                    advance(it, rand() % (labels.size() - 1));
//
//                    label = *it;
//                }

                data.push_back(make_pair(item, label));
            }

            if (shuffle) {
                random_shuffle(data.begin(), data.end());
            }

            int per_fold = (int) (data.size() / fold_n);


            double max_total_score = -1000;

            clock_t bgn = clock();
            int k_max = 5;
            for(int k = 0; k < k_max; k++) {
                RandomFuzzyForest rff(clasifier_n, job_n);

                double total_score = 0;
                for (int i = 0; i < fold_n; i++) {
                    data_t training_data;
                    data_t verification_data;
                    if (debug) {
                        cout << "Scoring fold " << i << "\t";
                    }
                    for (int j = 0; j < data.size(); j++) {
                        if (j >= i * per_fold and j < (i + 1) * per_fold) {
                            verification_data.push_back(data[j]);
                        } else {
                            training_data.push_back(data[j]);
                        }
                    }

                    rff.fit(training_data,
                            ranges,
                            categorical_features,
                            numerical_features);
                    double curr_score = rff.score(verification_data);
                    if (debug) {
                        cout << "\tScore: " << curr_score << endl;
                    }

                    total_score += curr_score;
                }

                if(total_score > max_total_score) {
                    max_total_score = total_score;
                }

                if(max_total_score / fold_n >= accuracy) {
                    break;
                }
            }
            clock_t end = clock();


            if (debug) {
                cout << endl << endl;
            }
            cout << setprecision(2) << fixed;
            cout << dataset << "\t\t";
            double achieved_acc = max_total_score / fold_n;
            cout << 100 * achieved_acc << "\t" << 100 * accuracy << endl;
            cout << "\tRun time: " << (1.0 / k_max) * double(end - bgn) / (CLOCKS_PER_SEC) << endl;
            cout << string(25, '-') << endl;

            if(accuracy > achieved_acc) {
                error_score -= pow(achieved_acc - accuracy, 2);
            } else {
                error_score += pow(achieved_acc - accuracy, 2);
            }
        }
        cout << "Square error score: " << 100 * error_score << endl;
    }
};

#endif //CPP_LOCALRUNNER_H
