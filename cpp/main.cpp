#include "includes.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"
#include "test_RandomForest.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
using namespace std;

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

int main() {
    bool shuffle = true;
    bool debug = false;
    bool run_tests = false;
    bool local = true;

    int clasifier_n = 10;

    int job_n = 4;

    int fold_n = 10;

    if (run_tests) {
        run_all_tests();

        return 0;
    }

    if (local) {
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
                                   "IMA",
                                   "THY",
                                   "WDB",
                                   "DER",
                                   "ION",
                                   "SON"};

//    datasets = {"HAB", "HAY", "TAE", "BUP"};
//        datasets = {"BUP"};

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
            for (auto &x : string_data) {
                vector<double> item;
                for (auto val : x.first) {
                    item.push_back(to_double(val));
                }
                string classification = x.second;
                data.push_back(make_pair(item, classification));
            }

            if (shuffle) {
                random_shuffle(data.begin(), data.end());
            }

            int per_fold = (int) (data.size() / fold_n);

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

            if (debug) {
                cout << endl << endl;
            }
            cout << setprecision(2) << fixed;
            cout << dataset << "\t\t";
            cout << total_score / fold_n << "\t" << accuracy << endl;
            cout << string(25, '-') << endl;
        }
    } else {
        

        return 0;
    }

    return 0;
}
#pragma clang diagnostic pop