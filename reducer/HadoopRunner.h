//
// Created by faruk on 4/30/17.
//

#ifndef CPP_HADOOPRUNNER_H
#define CPP_HADOOPRUNNER_H

#include "includes.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"

#define NONE_WORD "EMPTY"

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

template<typename Out>
void split_str(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split_str(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split_str(s, delim, std::back_inserter(elems));
    return elems;
}


class HadoopRunner {
    vector<int> attribute_inds;
    vector<int> numerical_features;
    vector<int> categorical_features;
    int class_ind;
    char separation_char;
    int line_cnt;
public:
    void run(int reducer_n,
             const string &dataset,
             int tree_n,
             string verification_data_path) {
        data_properties(dataset,
                        attribute_inds,
                        class_ind,
                        separation_char,
                        numerical_features,
                        categorical_features,
                        line_cnt);

        vector<RandomFuzzyForest> forests;

        int curr_id = -1;
        vector<example_t > curr_data;
        vector<example_t > curr_test_data;

        string line;
        while (getline(cin, line)) {
            trim(line);
            vector<string> tokens = split_str(line, '\t');

            string example_str = tokens[1];
            std::istringstream ss(tokens[0]);
            int id;
            ss >> id;

            if (id == curr_id) {
                example_t curr_example;

                if (example_str[0] == 't') {
                    curr_example = example_from_line(example_str.substr(1),
                                                     attribute_inds,
                                                     class_ind,
                                                     separation_char);

                    curr_test_data.push_back(curr_example);
                } else {
                    curr_example = example_from_line(example_str,
                                                     attribute_inds,
                                                     class_ind,
                                                     separation_char);

                    curr_data.push_back(curr_example);
                }
            } else {
                if (curr_id != -1) {
                    // build a tree for prev and classify
//                    train_and_predict(tree_n, curr_data, curr_test_data);
                }

                // start a new (off by one)
                curr_id = id;
                curr_data = vector<example_t >();
                curr_test_data = vector<example_t >();
            }
        }

        // take care of the last one
//        train_and_predict(tree_n, curr_data, curr_test_data);
    }

    void train_and_predict(int tree_n,
                           const vector<pair<vector<string>, string>> &curr_data,
                           const vector<pair<vector<string>, string>> &curr_test_data) const {
        vector<range_t > ranges;
        find_ranges(curr_data, ranges);

        data_t data;
        for (int i = 0; i < curr_data.size(); i++) {
            auto x = curr_data[i];
            cout << x << endl;
//            vector<double> item;
//            for (auto val : x.first) {
//                item.push_back(to_double(val));
//            }
//            string classification = x.second;
//            data.push_back(make_pair(item, classification));
        }

        RandomFuzzyForest rff(tree_n);
        rff.fit(data,
                ranges,
                categorical_features,
                numerical_features);
    }
};

#endif //CPP_HADOOPRUNNER_H
