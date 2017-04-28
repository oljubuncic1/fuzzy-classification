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
                                [i](example_t x, example_t y) -> double {
                                    return to_double(x.first[i]) < to_double(y.first[i]);
                                }
                        )->first[i]),
                        to_double(max_element(
                                examples.begin(),
                                examples.end(),
                                [i](example_t x, example_t y) -> double {
                                    return to_double(x.first[i]) < to_double(y.first[i]);
                                }
                        )->first[i])
                )
        );
    }
}

vector<int> generate_range(int n) {
    vector<int> rng = {};
    for(int i = 0; i < n; i++) {
        rng.push_back(i);
    }

    return rng;
}

void load_data(const string &dataset,
               vector<example_t > &data,
               vector<int> &categorical_features,
               vector<int> &numerical_features,
               double &accuracy) {
    if (dataset.compare("HAB") == 0) {
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
        numerical_features = {};
        categorical_features = {0, 1, 2, 3, 4};
        accuracy = 0.6255;
    } else if(dataset.compare("BUP") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/bupa.dat",
                             {0, 1, 2, 3, 4, 5},
                             6,
                             345);
        numerical_features = {5};
        categorical_features = {0, 1, 2, 3, 4};
        accuracy = 0.7220;
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
                             {1, 2, 3, 4, 5, 6, 7, 8},
                             9,
                             462);
        numerical_features = {0, 1, 2, 3, 6, 7, 8};
        categorical_features = {4};
        accuracy = 0.7051;
    } else if(dataset.compare("WIS") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/wisconsin.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8},
                             9,
                             699);
        numerical_features = {0, 1, 2, 3, 5, 6, 7, 8};
        categorical_features = {};
        accuracy = 0.9728;
    } else if(dataset.compare("CLE") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/cleveland.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                             13,
                             303);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        categorical_features = {};
        accuracy = 0.5836;
    } else if(dataset.compare("HEA") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/heart.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                             13,
                             270);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        categorical_features = {1, 2, 8};
        accuracy = 0.8389;
    } else if(dataset.compare("WIN") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/wine.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                             13,
                             178);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        categorical_features = {};
        accuracy = 0.9728;
    } else if(dataset.compare("AUS") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/australian.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                             14,
                             690);
        numerical_features = {};
        categorical_features = {0, 7, 8, 10};
        accuracy = 0.8600;
    } else if(dataset.compare("VEH") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/vehicle.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
                             18,
                             846);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        categorical_features = {};
        accuracy = 0.7538;
    } else if(dataset.compare("BAN") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/bands.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                             19,
                             539);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        categorical_features = {};
        accuracy = 0.7089;
    } else if(dataset.compare("HEP") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/hepatitis.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                             19,
                             155);
        numerical_features = {0, 13, 14, 15, 16, 17};
        categorical_features = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18};
        accuracy = 0.8958;
    } else if(dataset.compare("IMA") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/segmentation.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                             19,
                             2310);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        categorical_features = {};
        accuracy = 0.9620;
    } else if(dataset.compare("THY") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/thyroid.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
                             21,
                             7200);
        numerical_features = {0, 16, 17, 18, 19};
        categorical_features = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        accuracy = 0.9730;
    } else if(dataset.compare("WDB") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/wdbc.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
                             30,
                             569);
        numerical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
        categorical_features = {};
        accuracy = 0.9601;
    } else if(dataset.compare("DER") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/dermatology.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33},
                             34,
                             366);
        numerical_features = {33};
        categorical_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
        accuracy = 0.9764;
    } else if(dataset.compare("ION") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/ionosphere.dat",
                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
                             33,
                             351);
        numerical_features = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
        categorical_features = {0};
        accuracy = 0.9201;
    } else if(dataset.compare("SON") == 0) {
        data = load_csv_data("/home/faruk/workspace/thesis/data/sonar.dat",
                             generate_range(60),
                             60,
                             208);
        numerical_features = generate_range(60);
        categorical_features = {};
        accuracy = 0.7993;
    }
}

#endif