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

using namespace std;

#define example_t pair<vector<string>, string>
#define range_t pair<double, double>

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
        example_t e = example_from_line(string(line),
                                        attribute_indices,
                                        class_index,
                                        separation_char);
        data.push_back(e);
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

vector<range_t > find_ranges(
        const vector<example_t > &examples
) {
    vector<int> indices = get_range((const int &) examples[0].first.size());
    vector<range_t > ranges;
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

    return ranges;
}

void load_data(const string &dataset,
               vector<example_t > &data,
               vector<int> &categorical_features,
               vector<int> &numerical_features) {
    if (dataset.compare("AUS") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/cpp/data/australian.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                             14,
                             690,
                             ' ');
    } else if (dataset.compare("HAB") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/cpp/data/australian.dat",
                             {0, 1, 2},
                             14,
                             306,
                             ' ');
        categorical_features = {};
        numerical_features = {0, 1, 2};
    } else if (dataset.compare("SEG") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/segmentation.dat",
                             {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                             0,
                             2310);
    }
}

#endif