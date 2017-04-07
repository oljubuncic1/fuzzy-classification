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
#define example_data_t vector<string>

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
	const int &class_index
) {
	auto parts = split(line, ',');
	string classification = parts[ class_index ];

	vector<string> data;
	for(auto i : attribute_indices) {
		data.push_back(parts[i]);
	}

	example_t example = make_pair(data, classification);
	return example;
}

vector<example_t> load_csv_data(
	const string &file_name, 
	const vector<int> &attribute_indices, 
	const int &class_index, 
	const int &line_cnt
) {
	auto in = fopen(file_name.c_str(), "r");

    vector<example_t> data;

	char line[1000];
	for(int i = 0; i < line_cnt; i++) {
		fscanf(in, "%s", line);
		example_t e = example_from_line(string(line), attribute_indices, class_index);
		// if(e.second.compare("0") == 0 or e.second.compare("1") == 0) {
			data.push_back(e);
		// }
	}

	fclose(in);

    return data;
}

vector<range_t> find_ranges(
	const vector<example_t> &examples, 
	const vector<int> &indices
) {
	vector<range_t> ranges;
	for(auto i : indices) {
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

#endif