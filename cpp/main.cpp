#include "includes.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"

using namespace std;


int main() {
    bool shuffle = true;
    bool debug = false;

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
                               "SAH"};

    datasets = {"SAH"};

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

        int clasifier_n = 10;
        int job_n = 4;

        int fold_n = 10;
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

    return 0;
}