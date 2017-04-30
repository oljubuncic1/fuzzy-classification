
#ifndef CPP_TEST_RANDOMFOREST_H
#define CPP_TEST_RANDOMFOREST_H

#include "includes.h"
#include "RandomFuzzyTree.h"

#include <cassert>

void test_all_numeric() {
    function<vector<int>(void)> rfg = []() -> vector<int> {
        static int i = 0;

        vector<vector<int>> feature_sets = {
                {0, 1},
        };

        return feature_sets[i++];
    };

    RandomFuzzyTree rft;
    rft.set_random_feature_generator(rfg);
    rft.set_max_depth(0);

    data_t data = {
            { {0.1, 0.1, 3, 4}, "1" },
            { {0.2, 0.6, 3, 4}, "1" },
            { {0.3, 1.2, 3, 4}, "1" },
            { {0.6, 0.2, 3, 4}, "2" },
            { {0.7, 0.7, 3, 4}, "2" },
            { {0.8, 1.3, 3, 4}, "2" },
            { {1.2, 0.3, 3, 4}, "3" },
            { {1.3, 0.8, 3, 4}, "3" },
            { {1.4, 1.4, 3, 4}, "3" },
    };
    vector<range_t> ranges = {
            {0, 1.4},
            {0, 1.4},
    };

    vector<int> categorical_features = {};
    vector<int> numerical_features = {0, 1};

    rft.fit(data,
            ranges,
            categorical_features,
            numerical_features);

    assert(rft.root.feature == 0);
    assert(eq(rft.root.cut_point, 0.6));

    item_t dmy = { {0, 2, 3, 4}, "1" };
    item_t dmy2 = { {0, 2, 3, 4}, "1" };

    assert(eq(rft.root.children[0]->f(dmy), 1));

    dmy.first[0] = 0.6;
    assert(eq(rft.root.children[0]->f(dmy), 0));

    dmy.first[0] = -1;
    assert(eq(rft.root.children[0]->f(dmy), 0));

    dmy.first[0] = 0.7;
    assert(eq(rft.root.children[0]->f(dmy), 0));

    dmy.first[0] = 0.3;
    dmy2.first[0] = 0.4;
    assert(rft.root.children[0]->f(dmy2) < rft.root.children[0]->f(dmy));

    dmy.first[0] = 0;
    assert(eq(rft.root.children[1]->f(dmy), 0));

    dmy.first[0] = 0.6;
    assert(eq(rft.root.children[1]->f(dmy), 1));

    dmy.first[0] = 1.4;
    assert(eq(rft.root.children[1]->f(dmy), 0));

    dmy.first[0] = 0.3;
    dmy2.first[0] = 0.4;
    assert(rft.root.children[1]->f(dmy) < rft.root.children[1]->f(dmy2));

    dmy.first[0] = 0.7;
    dmy2.first[0] = 0.8;
    assert(rft.root.children[1]->f(dmy2) < rft.root.children[1]->f(dmy));

    dmy.first[0] = 0.8;
    dmy2.first[0] = 0.9;
    assert(rft.root.children[1]->f(dmy2) < rft.root.children[1]->f(dmy));
}

void run_all_tests() {
    test_all_numeric();

    cout << "All tests passed" << endl;
}

#endif //CPP_TEST_RANDOMFOREST_H
