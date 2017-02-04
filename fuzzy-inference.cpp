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

using namespace std;

#define fdd function<double(const double)>
#define SPECIAL_VALUE 225883.225883

bool eq(double x, double y) {
	return fabs(x - y) <= 1e-7;
}

fdd triangular(double center, double width, double name = SPECIAL_VALUE + 1) {
	return [center, width, name](const double x) {
		if(eq(x, SPECIAL_VALUE)) {
			return name;
		}
		double r = width / 2;
		double k = 1 / r;
		
		double left = center - r;
		double right = center + r;

		if(left <= x <= center) {
			return k * (x - left) + 0.0;
		} else if (center <= x <= right) {
			return -k * (x - center) + 1.0;
		} else {
			return 0.0;
		}
	};
}

#define range_t pair<double, double>

vector<fdd> labels(const range_t &rng, int label_count) {
	double L = rng.second - rng.first;
	double w = 2.0 * L / (label_count - 1);

	if(w == 0) {
		w = 1; // hacky, because division by zero
	}

	vector<function<double(double)>> lbls;
	for(int i = 0; i < label_count; i++) {
		lbls.push_back(
			triangular( rng.first + (i) * w / 2, w, i )
		);
	}

	return lbls;
}

#define db_t vector<vector<fdd>>

db_t generate_db(const vector<range_t> &ranges,  int label_count) {
	vector<vector<function<double(double)>>> db;
	for(auto r : ranges) {
		db.push_back(labels(r, label_count));
	}
	return db;
}

#define example_t pair<vector<string>, string>

string classification(const example_t &example) {
	return example.second;
}

vector<string> input_data(const example_t &example) {
	return example.first;
}

#define example_data_t vector<string>
#define rule_t tuple<vector<fdd>, string, double>

double to_double(const string &str) {
	return stod(str);
}

double matching_degree(const example_data_t &example_data, const rule_t &x) {
	double dgr = 1.0;
	for(int i = 0; i < example_data.size(); i++) {
		dgr *= get<0>(x)[i](to_double(example_data[i]));
	}
	return dgr;
}

rule_t rule(const example_t &example, const db_t &db) {
	// cout << "\t" << db.size() << endl;
	// cout << "\t" << example.first.size() << endl;
	// cin.get();

	auto data = input_data(example);

	double rw = 0.1;
	auto rule =  make_tuple(vector<fdd>(), classification(example), rw);


	for(int i = 0; i < db.size(); i++) {
		// cout << "\t\t" << i << endl;
		// cout << "\t\t" <<db[i].size() << endl;
		// cin.get();
		// auto max_f = db[i][0];
		// for(int j = 0; j < db[i].size(); j++) {
		// 	auto f = db[i][j];
		// 	if(f(to_double(data[i])) > max_f(to_double(data[i]))) {
		// 		max_f = f;
		// 	}
		// }
		// get<0>(rule).push_back(max_f);
		get<0>(rule).push_back(
			*max_element(
				db[i].begin(),
				db[i].end(),
				[ data, i ](fdd x, fdd y) { 
					return x(to_double(data[i])) < y(to_double(data[i])); 
				}
			)
		);
	}

	return rule;
}

double weight(rule_t &rule, vector<example_t> examples) {
	vector<example_t> examples_in_class(examples.size());
	auto it = copy_if(
		examples.begin(), 
		examples.end(), 
		examples_in_class.begin(),
		[rule](example_t e) { 
			return e.second == get<1>(rule); 
		}
	);
	examples_in_class.resize(it - examples.begin());

	vector<example_t> examples_out_of_class(examples.size());
	it = copy_if(
		examples.begin(), 
		examples.end(), 
		examples_out_of_class.begin(),
		[rule](example_t e) { 
			return e.second != get<1>(rule); 
		}
	);
	examples_out_of_class.resize(it - examples.begin());

	double in_class_md = 0; 
	for(auto &e : examples_in_class) {
		in_class_md += matching_degree(e.first, rule);
	}

	double out_of_class_md = 0; 
	for(auto &e : examples_in_class) {
		out_of_class_md += matching_degree(e.first, rule);
	}

	if(in_class_md == 0 and out_of_class_md == 0) {
		return 0;
	}

	double weight = (in_class_md - out_of_class_md) / (in_class_md + out_of_class_md);

	return weight;
}

#define rb_t vector<rule_t>

void update_weights(rb_t &rb, vector<example_t> &examples) {
	for(auto r : rb) {
		get<2>(r) = weight(r, examples);
	}
}

void update_weight(rule_t &r, vector<example_t> &examples) {
	get<2>(r) = weight(r, examples);
}

string rule_desc(rule_t rule) {
	string rule_desc_str = "";

	for(auto &f : get<0>(rule)) {
		rule_desc_str += f(SPECIAL_VALUE);
	}
	rule_desc_str += get<2>(rule);

	return rule_desc_str;
}

void remove_duplicates(rb_t &rb) {
	map<string, vector<rule_t>> rule_map;
	for(auto &r : rb) {
		rule_map[rule_desc(r)].push_back(r);
	}

	rb_t best_rules;
	
	for(auto &kv : rule_map) {
		best_rules.push_back(
			*max_element(
				kv.second.begin(),
				kv.second.end(), 
				[](rule_t x, rule_t y) { return get<2>(x) < get<2>(y); }
			)
		);
	}

	rb = best_rules;
}

void print_perc(int perc) {
	for(int i = 0; i < 100; i++) {
		if(i <= perc) {
			cout << '|';//(char)(178);
		} else {
			cout << " ";
		}

		if(i == 50) {
			cout << " " << perc << "% ";
		}
	}
	cout << endl;
}

rb_t generate_rb(vector<example_t> &examples, const db_t &db) {
	rb_t rb;
	for(auto &e : examples) {
		rb.push_back(rule(e, db));
	}

	system("clear");
	cout << "Updating weights..." << endl;
	
	#define THREAD_NUM 4
	#define PERCS 5

	int last_perc = 0;
	print_perc(0);
	for(int j = 0; j < rb.size() / THREAD_NUM; j++) {
		thread threads[THREAD_NUM];
		for(int i = 0; i < THREAD_NUM; i++) {
		    threads[i] = thread(update_weight, ref(rb[THREAD_NUM *j + i]), ref(examples));
		}

		for(int i = 0; i < THREAD_NUM; i++) {
		    threads[i].join();
		}

		int perc = PERCS * (int(100.0 * (4.0 * j) / rb.size()) / PERCS);
		if(perc > last_perc) {
			system("clear");
			cout << "Updating weights..." << endl;
			print_perc(perc);
		}
		last_perc = perc;
	}

	system("clear");
	cout << "Updating weights..." << endl;
	print_perc(100);

	// update_weights(rb, examples);

	cout << "Removing duplicates..." << endl;
	remove_duplicates(rb);

	return rb;
}
	

void classify(example_t &example, rb_t &rb, string &my_class) {
	rule_t max_rule = *max_element(
		rb.begin(),
		rb.end(), 
		[example](rule_t x, rule_t y) { 
			return (matching_degree(example.first, x) * get<2>(x)) < 
				(matching_degree(example.first, y) * get<2>(y)); 
		}
	);

	my_class = get<1>(max_rule);
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

example_t example_from_line(const string &line, const vector<int> &attribute_indices, const int &class_index) {
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
		data.push_back(example_from_line(string(line), attribute_indices, class_index));
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
						return x.first[i] < y.first[i];
					}
				)->first[i]),
				to_double(max_element(
					examples.begin(), 
					examples.end(), 
					[i](example_t x, example_t y) {
						return x.first[i] < y.first[i];
					}
				)->first[i])
			)
		);
	}

	// *min_element([float(e[0][i]) for e in examples]),
	// 			*max_element([float(e[0][i]) for e in examples])
	return ranges;
}

// def print_rb(rb, rw_pos = 3):
// 	for rule in rb:
// 		rule_str = ''
// 		i = 0
// 		for f in rule[0]:
// 			rule_str +=  'x' + str(i) + ' is ' + f('name') + '   '
// 			i += 1
// 		rule_str += " class is " + str(rule[1]) + " with rw " + str(round(rule[2], rw_pos))
// 		print(rule_str)

vector<int> num_range(const int &n) {
	vector<int> nums;
	for(int i = 0; i < n; i++) {
		nums.push_back(i);
	}

	return nums;
}

void print_data(const vector<example_t> &examples) {
	for(auto example : examples) {
		for(int i = 0; i < example.first.size(); i++) {
			cout << example.first[i] << " ";
		}
		cout << " -> ";
		cout << example.second << endl;
	}
}

vector<example_t> load_data(const vector<int> &cols) {
	auto data = load_csv_data(
		"kddcupfull.data", 
		cols,
		41,
		2000000
	);
	
	std::srand ( unsigned ( std::time(0) ) );
	random_shuffle(data.begin(), data.end());

	data.resize(10000);

	return data;
}

int main() {
	ios::sync_with_stdio(false);
    cin.tie(NULL);

    system("clear");

	cout << "Loading data..." << endl;

	vector<int> cols = {0, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 
		30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
	auto data = load_data(cols);

	cout << "Data loaded" << endl;

	// random.shuffle(data)
	// data = data[-2000:]

	// # for item in data:
	// # 	if item[1] != 'normal.':
	// # 		item[1] = 'ddos.'

	// print_data(data);

	double validation_data_perc = 0.1;
	int validation_examples = int(validation_data_perc * data.size());

	vector<example_t> training_data(data.size() - validation_examples);
	copy(data.begin(), data.end() - validation_examples, training_data.begin());

	vector<example_t> verification_data(validation_examples);
	copy(data.end() - validation_examples, data.end(), verification_data.begin());

	cout << "Generating db..." << endl;
	int label_count = 3;

	auto ranges = find_ranges(data, num_range(cols.size()));
	auto db = generate_db(ranges, label_count);

	cout << "Generating rb..." << endl;
	auto rb = generate_rb(training_data, db);

	cout << "Classifying..." << endl;

	vector<string> classifications(validation_examples);

	// thread threads[THREAD_NUM];		
	// for(int j = 0; j < verification_data.size() / THREAD_NUM; j++) {

	// 	for(int i = 0; i < THREAD_NUM; i++) {
	// 		auto v = verification_data[THREAD_NUM * j + i];
	// 		threads[i] = thread(classify, ref(v), ref(rb), ref(classifications[THREAD_NUM * j + i]));
	// 	}

	// 	for(int i = 0; i < THREAD_NUM; i++) {
	// 		threads[i].join();
	// 	}
	// }

	for(int i = 0; i < classifications.size(); i++) {
		classify(verification_data[i], rb, classifications[i]);
	}

	int total = 0;
	for(auto i : num_range(verification_data.size())) {
		auto verification = verification_data[i];
		cout << classifications[i] << " " << verification.second << endl;
		if(classifications[i] == verification.second) {
			total++;
		}
	}

	cout << "Accuracy% " <<  (100.0 * total / verification_data.size() ) << endl;

	// for c in set( [ x[1] for x in verification_data ] ):
	// 	print(
	// 		c + " " + str( len( [x for x in verification_data if x[1] == c] ) )
	// 	)

	return 0;
}
