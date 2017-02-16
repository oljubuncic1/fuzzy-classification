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

#define fdd function<double(const double)>
#define SPECIAL_VALUE 225883.225883
#define PRINT_VALUE 225884.225883
#define PRINT_NAME 235884.225883
#define THREAD_NUM 4

#define range_t pair<double, double>
#define db_t vector<vector<fdd>>
#define example_t pair<vector<string>, string>
#define example_data_t vector<string>
#define rule_t tuple<vector<fdd>, string, double>
#define rb_t vector<rule_t>

template<typename T>
ostream& operator<<(ostream &os, const vector<T> &v) {
	for(int i = 0; i < v.size(); i++) {
		os << v[i] << " ";
	}

	return os;
}

template<typename U, typename V>
ostream& operator<<(ostream &os, const pair<U, V> &p) {
    os << "(" << p.first << ", " << p.second << ")";

    return os;
}

template<typename U, typename V>
ostream& operator<<(ostream &os, const map<U, V> &m) {
    for(auto kv : m) {
        cout << kv.first << " --> " << kv.second << endl;
    }

    return os;
}

bool eq(double x, double y) {
	return fabs(x - y) <= 1e-7;
}

void print_rule(const rule_t &r) {
	int i = 0;
	for(auto f : get<0>(r)) {
		f(PRINT_NAME);
		if(i != get<0>(r).size() - 1) {
			cout << "|";
		} else {
			cout << " -> " << get<1>(r) << " rw = " << get<2>(r) << endl;
		}
		i++;
	}
}

void print_rb(rb_t rb) {
	for(auto r : rb) {
		print_rule(r);
	}
}

fdd semicircle(double center, double width, double name = SPECIAL_VALUE + 1) {
	return [center, width, name](const double x) {
		double r = width / 2;
		double k = 1 / r;

		double left = center - r;
		double right = center + r;

		if(eq(x, SPECIAL_VALUE)) {
			return name;
		} else if(eq(x, PRINT_VALUE)) {
			cout << left << " to " << right << " k = " << k << " center: " << center;
			return 0.0;
		} else if(eq(x, PRINT_NAME)) {
			cout << name;
			return 0.0;
		}
		
		double val = r * r - (x - center) * (x - center);
		if(val <= 0) {
			return 0.0;
		} else {
			return (1.0 / r) * sqrt(val);
		}
	};
}

fdd triangular(double center, double width, double name = SPECIAL_VALUE + 1) {
	return [center, width, name](const double x) {
		double r = width / 2;
		double k = 1 / r;

		double left = center - r;
		double right = center + r;

		if(eq(x, SPECIAL_VALUE)) {
			return name;
		} else if(eq(x, PRINT_VALUE)) {
			cout << left << " to " << right << " k = " << k << " center: " << center;
			return 0.0;
		} else if(eq(x, PRINT_NAME)) {
			cout << name;
			return 0.0;
		}

		if(left <= x and x <= center) {
			return k * (x - left) + 0.0;
		} else if (center <= x and x <= right) {
			return -1 * k * (x - center) + 1.0;
		} else {
			return 0.0;
		}
	};
}

vector<fdd> labels(const range_t &rng, int label_count) {
	double L = rng.second - rng.first;
	double w = 2.0 * L / (label_count - 1);

	if(w == 0) {
		w = 1; // hacky, because division by zero
	}

	vector<function<double(double)>> lbls;
	for(int i = 0; i < label_count; i++) {
		lbls.push_back(
			triangular( rng.first + (i) * w / 2.0, w, i )
		);
	}

	return lbls;
}

void generate_labels(db_t *db, int from, int to, vector<range_t> ranges, int label_count) {
	for(int i = from; i < to; i++) {
		(*db)[i] = labels(ranges[i], label_count);
	}
}

db_t generate_db(vector<range_t> ranges,  int label_count) {
	vector<vector<function<double(double)>>> db(ranges.size());
	
	thread threads[THREAD_NUM];
	int per_thread = db.size() / THREAD_NUM;
	for(int i = 0; i < THREAD_NUM; i++) {
		int end;
		if(i == THREAD_NUM - 1) {
			end = db.size();
		} else {
			end = (i + 1) * per_thread;
		}
		threads[i] = thread(generate_labels, &db, i * per_thread, end, ranges, label_count);
	}

	for(int i = 0; i < THREAD_NUM; i++) {
		threads[i].join();
	}

	return db;

	for(auto r : ranges) {
		db.push_back(labels(r, label_count));
	}
	return db;
}

string classification(const example_t &example) {
	return example.second;
}

vector<string> input_data(const example_t &example) {
	return example.first;
}

double to_double(const string &str) {
	return stod(str);
}

double matching_degree(const example_data_t &example_data, const rule_t &x) {
	double dgr = 1.0;
	for(int i = 0; i < example_data.size(); i++) {
		auto f = get<0>(x)[i];
		double curr = f(to_double(example_data[i]));

		if(eq(curr, 0)) {
			return 0;
		}
		
		dgr *= curr;
	}
	return dgr;
}

rule_t rule(const example_t &example, const db_t &db) {
	auto data = input_data(example);

	double rw = 0.1;
	auto rule =  make_tuple(vector<fdd>(), classification(example), rw);

	for(int i = 0; i < db.size(); i++) {
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

	get<2>(rule) = matching_degree(data, rule);
	return rule;
}

void print_example(const example_t &example) {
	for(int i = 0; i < example.first.size(); i++) {
		cout << example.first[i] << " ";
	}
	cout << " -> ";
	cout << example.second << endl;
}

double weight(rule_t rule, vector<example_t> examples) {
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
		double curr_md = matching_degree(e.first, rule);
		in_class_md += curr_md;
	}

	double out_of_class_md = 0; 
	for(auto &e : examples_in_class) {
		out_of_class_md += matching_degree(e.first, rule);
	}

	if(eq(in_class_md, 0) and eq(out_of_class_md, 0)) {
		return 0;
	}

	double weight = 1.0 * (in_class_md - out_of_class_md) / (in_class_md + out_of_class_md);

	return weight;
}

void update_weights(rb_t *rb, int from, int to, vector<example_t> examples) {
	for(int i = from; i < to; i++) {
		auto r = (*rb)[i];
		double next_weight = weight(r, examples);
		get<2>(r) = next_weight;
	}
}

void update_weight(rule_t *r, vector<example_t> examples) {
	double next_weight = weight(*r, examples); 
	get<2>(*r) = next_weight;
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
	
	for(auto kv : rule_map) {
		// best_rules.push_back(
		// 	*max_element(
		// 		kv.second.begin(),
		// 		kv.second.end(), 
		// 		[](rule_t x, rule_t y) { return get<2>(x) < get<2>(y); }
		// 	)
		// );
		
		for(auto r : kv.second) {

		}
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

string to_string(const rule_t &r) {
	string r_str = "";
	for(auto f : get<0>(r)) {
		r_str += f(SPECIAL_VALUE) + '0';
	}

	return r_str;
}

rb_t generate_rb(vector<example_t> &examples, const db_t &db) {
	rb_t rb;

	// map<string, vector<example_t>> rule_base;
	// for(auto e : examples) {
	// 	auto r = rule(e, db);
	// 	rule_base[to_string(r)].push_back(e);
	// }

	// int max_len = 0;
	// vector<example_t> max_rule;
	// for(auto kv : rule_base) {
	// 	// cout << kv.first << " -> ";
	// 	if(kv.second.size() > max_rule.size()) {
	// 		max_rule = kv.second;
	// 	}
	// 	for(auto v : kv.second) {
	// 		// cout << v;
	// 	}
	// 	cout << endl;
	// }
	// cout << max_rule.size() << endl;
	// for(auto v : max_rule) {
	// 	cout << v;
	// }
	// cin.get();

	int i = 0;
	for(auto &e : examples) {
		rb.push_back(rule(e, db));
	}

	cout << "Updating weights..." << endl;

	thread threads[THREAD_NUM];
	int per_thread = rb.size() / THREAD_NUM;

	for(int i = 0; i < THREAD_NUM; i++) {
		int end;
		if(i == THREAD_NUM - 1) {
			end = rb.size();
		} else {
			end = (i + 1) * per_thread;
		}
		threads[i] = thread(update_weights, &rb, i * per_thread, end ,examples);
	}

	for(int i = 0; i < THREAD_NUM; i++) {
	    threads[i].join();
	}

	cout << "Removing duplicates..." << endl;
	remove_duplicates(rb);

	return rb;
}
	
string classify(example_t example, rb_t rb) {
	rule_t max_rule = *max_element(
		rb.begin(),
		rb.end(), 
		[example](rule_t x, rule_t y) { 
			return (matching_degree(example.first, x) * get<2>(x)) < 
				(matching_degree(example.first, y) * get<2>(y)); 
		}
	);

	if(eq(get<2>(max_rule), 0)) {
		cout << "Not sure..." << endl;
	}

	return get<1>(max_rule);
}

void classify_range(
	vector<example_t> *data, 
	int from, 
	int to, 
	rb_t rb
) {
	for(int i = from; i < to; i++) {
		string my_class;
		(*data)[i].second = classify( (*data)[i], rb );
	}
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

vector<int> num_range(const int &n) {
	vector<int> nums;
	for(int i = 0; i < n; i++) {
		nums.push_back(i);
	}

	return nums;
}

void print_data(const vector<example_t> &examples) {
	for(auto example : examples) {
		print_example(example);
	}
}

vector<example_t> load_data(
	const string &file_name, 
	const vector<int> &cols, 
	const int &class_col, 
	const int &rows, 
	const int &data_sz
) {
	auto data = load_csv_data(
		file_name, 
		cols,
		class_col,
		rows
	);
	
	std::srand ( unsigned ( std::time(0) ) );
	// random_shuffle(data.begin(), data.end());

	data.resize(data_sz);

	return data;
}

void print_ranges(const vector<range_t> &ranges) {
	int i = 0;
	for(auto r : ranges) {
		cout << (i++) << " -> (" << r.first << ", " << r.second << ")" << endl;
	}
}

pair< vector<example_t>, vector<int> > load_poker_data() {
	vector<int> cols = num_range(10);
	return make_pair(
		load_data("poker-hand-testing.data", cols, 10, 1000000, 1000),
		cols
	);
}

pair< vector<example_t>, vector<int> > load_kddcup_data() {
	vector<int> cols = {0, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 
		30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};	
	return make_pair(
		load_data("kddcupfull.data", cols, 41, 2000000, 1000),
		cols
	);
}

pair< vector<example_t>, vector<int> > load_iris_data() {
	vector<int> cols = num_range(4);	
	return make_pair(
		load_data("iris.dat", cols, 4, 150, 150),
		cols
	);
}

pair< vector<example_t>, vector<int> > load_titanic_data() {
	vector<int> cols = num_range(3);	
	return make_pair(
		load_data("takanik.dat", cols, 3, 2201, 2201),
		cols
	);
}  

int main() {
	ios::sync_with_stdio(false);
    cin.tie(NULL);
    system("clear");

	cout << "Loading data..." << endl;
	auto data_cols = load_poker_data();
	auto data = data_cols.first;
	auto cols = data_cols.second;
	cout << "Data loaded" << endl;

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
	vector<example_t> classifications(verification_data);

	thread threads[THREAD_NUM];		
	int per_thread = verification_data.size() / THREAD_NUM;

	for(int i = 0; i < THREAD_NUM; i++) {
		int end;
		if(i == THREAD_NUM - 1) {
			end = classifications.size();
		} else {
			end = (i + 1) * per_thread;
		}
		threads[i] = thread(classify_range, &classifications, i * per_thread, end, rb);
	}

	for(int i = 0; i < THREAD_NUM; i++) {
		threads[i].join();
	}

	int total = 0;
	for(auto i : num_range(verification_data.size())) {
		if(classifications[i].second == verification_data[i].second) {
			total++;
		}
	}

	cout << "Accuracy% " <<  (100.0 * total / verification_data.size() ) << endl;

	map<string, int> distribution;

	for(auto x : verification_data) {
		distribution[x.second]++;
	}

	for(auto m : distribution) {
		cout << m.first << " " << m.second << endl;
	}

	return 0;
}
