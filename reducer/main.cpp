#include "includes.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"
#include "test_RandomForest.h"
#include "LocalRunner.h"
#include "HadoopRunner.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"

using namespace std;

int main() {
    bool shuffle = true;
    bool debug = false;
    bool run_tests = false;
    bool local = false;

    int reducer_n = 3;
    string verification_data_path = "./testdat";
    string big_dataset = "WIS";

    int tree_n = 10;

    int clasifier_n = 100;

    int job_n = 4;

    int fold_n = 10;

    if (run_tests) {
        run_all_tests();

        return 0;
    }

    if (local) {
        LocalRunner localRunner;
        localRunner.run(shuffle, debug, clasifier_n, job_n, fold_n);
    } else {
        // reducer code
        HadoopRunner hadoopRunner;
        hadoopRunner.run(reducer_n, big_dataset, tree_n, verification_data_path);

        return 0;
    }

    return 0;
}

#pragma clang diagnostic pop