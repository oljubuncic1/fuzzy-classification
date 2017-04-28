#ifndef _DATA_H
#define _DATA_H

#include <functional>
#include <vector>
#include <string>
#include <tuple>
#include <utility> // std::pair
#include <cmath>
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thread>
#include <cstdio>
#include <atomic>
#include "definitions.h"

using namespace std;


double to_double(const string &str) {
    return stod(str);
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

example_t example_from_line(
        const string &line,
        const vector<int> &attribute_indices,
        const int &class_index,
        const char &separation_char
) {
    auto parts = split(line, separation_char);
    string classification = parts[class_index];

    vector<string> data;
    for (auto i : attribute_indices) {
        data.push_back(parts[i]);
    }

    example_t example = make_pair(data, classification);
    return example;
}

vector<example_t > load_csv_data(
        const string &file_name,
        const vector<int> &attribute_indices,
        const int &class_index,
        const int &line_cnt,
        const char &separation_char = ','
) {
    auto in = fopen(file_name.c_str(), "r");

    vector<example_t > data;

    char line[1000];
    for (int i = 0; i < line_cnt; i++) {
        fgets(line, 1000, in);
        if(string(line).find("?") == string::npos) {
            example_t e = example_from_line(string(line),
                                            attribute_indices,
                                            class_index,
                                            separation_char);
            data.push_back(e);
        }
    }

    fclose(in);

    return data;
}

vector<int> get_range(const int &n) {
    vector<int> rng;
    for (int i = 0; i < n; i++) {
        rng.push_back(i);
    }

    return rng;
}

void find_ranges(
        const vector<example_t > &examples,
        vector<range_t> &ranges
) {
    ranges = {};
    vector<int> indices = get_range((const int &) examples[0].first.size());
    for (auto i : indices) {
        ranges.push_back(
                make_pair(
                        to_double(min_element(
                                examples.begin(),
                                examples.end(),
                                [i](example_t x, example_t y) {
                                    return to_double(x.first[i]) < to_double(y.first[i]);
                                }
                        )->first[i]),
                        to_double(max_element(
                                examples.begin(),
                                examples.end(),
                                [i](example_t x, example_t y) {
                                    return to_double(x.first[i]) < to_double(y.first[i]);
                                }
                        )->first[i])
                )
        );
    }
}

void load_data(const string &dataset,
               vector<example_t > &data,
               vector<int> &categorical_features,
               vector<int> &numerical_features,
               double &accuracy) {
    if (dataset.compare("AUS") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/cpp/data/australian.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                             14,
                             690,
                             ' ');
    } else if (dataset.compare("HAB") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/haberman.dat",
                             {0, 1, 2},
                             3,
                             306);
        numerical_features = {0, 1, 2};
        categorical_features = {};
        accuracy = 0.7072;
    } else if (dataset.compare("SEG") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/segmentation.dat",
                             {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                             0,
                             2310);
    } else if(dataset.compare("HAY") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/hayes-roth.dat",
                             {2, 3, 4},
                             5,
                             132);
        numerical_features = {};
        categorical_features = {0, 1, 2};
        accuracy = 0.8082;
    } else if(dataset.compare("IRI") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/iris.dat",
                             {0, 1, 2, 3},
                             4,
                             150);
        numerical_features = {0, 1, 2, 3};
        categorical_features = {};
        accuracy = 0.9533;
    } else if(dataset.compare("MAM") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/mamographic.dat",
                             {0, 1, 2, 3, 4},
                             5,
                             961);
        numerical_features = {1};
        categorical_features = {0, 2, 3, 4};
        accuracy = 0.8380;
    } else if(dataset.compare("NEW") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/new_thyroid.dat",
                             {1, 2, 3, 4, 5},
                             0,
                             215);
        numerical_features = {0, 1, 2, 3, 4};
        categorical_features = {};
        accuracy = 0.9727;
    } else if(dataset.compare("TAE") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/tae.dat",
                             {0, 1, 2, 3, 4},
                             5,
                             151);
        numerical_features = {4};
        categorical_features = {0, 1, 2, 3};
        accuracy = 0.6255;
    } else if(dataset.compare("BUP") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/bupa.dat",
                             {0, 1, 2, 3, 4, 5},
                             6,
                             345);
        numerical_features = {0, 1, 2, 3, 4, 5};
        categorical_features = {};
        accuracy = 0.7270;
    } else if(dataset.compare("APP") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/appendicitis.dat",
                             {0, 1, 2, 3, 4, 5, 6},
                             7,
                             106);
        numerical_features = {0, 1, 2, 3, 4, 5, 6};
        categorical_features = {};
        accuracy = 0.8773;
    } else if(dataset.compare("PIM") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/pima.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7},
                             8,
                             768);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7};
        categorical_features = {};
        accuracy = 0.7648;
    } else if(dataset.compare("GLA") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/glass.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                             10,
                             214);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        categorical_features = {};
        accuracy = 0.7513;
    } else if(dataset.compare("SAH") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/saheart.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                             10,
                             462);
        numerical_features = {0, 1, 2, 3, 5, 6, 7, 8, 9};
        categorical_features = {4};
        accuracy = 0.7051;
    } else if(dataset.compare("WIS") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/winsconsin.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                             10,
                             462);
        numerical_features = {0, 1, 2, 3, 5, 6, 7, 8, 9};
        categorical_features = {4};
        accuracy = 0.7051;
    }
}

#endif