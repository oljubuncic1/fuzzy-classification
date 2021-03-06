#include <iostream>
#include <string>

#include <set>
#include <functional>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thread>
#include <cstdio>
#include <queue>
#include <thread>
#include <iomanip>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <iostream>
#include <string>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <sstream>

#include <functional>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
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

void produce_output(string &curr_example, map<string, double> &curr_predictions) {
    if(curr_example != "") {
        // finish last one

        trim(curr_example);
        vector<string> tokens = split_str(curr_example, ',');
        string correct_label = *(tokens.end() - 1);
        correct_label = correct_label.substr(0, correct_label.length() - 1);

        trim(correct_label);

        string max_label = "";
        double max_membership = 0;
        for(auto &kv : curr_predictions) {
            if(kv.second > max_membership) {
                max_membership = kv.second;
                max_label = kv.first;
            }
        }    
        trim(max_label);

        if(correct_label == max_label) {
            cout << "1" << endl;
        }
    }
}

int main(int argc, char **argv) {
    string curr_example = "";
    map<string, double> curr_predictions;

    string line;
    int i = 0;
	while (getline(cin, line)) {
        trim(line);
        vector<string> tokens = split_str(line, '\t');

        string example_str = tokens[0];
        string predictions_str = tokens[1];

        vector<string> predictions = split_str(predictions_str, ' ');

        predictions.erase(
            remove_if(predictions.begin(), predictions.end(), [](string s) { return s == " " or s == ""; }),
            predictions.end()
        );

        if(curr_example == example_str) {
            // continue
            for(int i = 0; i < predictions.size() / 2; i++) {
                curr_predictions[predictions[2*i]] += 
                    stod(predictions[2*i + 1]);
            }
        } else {
            produce_output(curr_example, curr_predictions);

            curr_example = example_str;
            curr_predictions = map<string, double>();
            for(int i = 0; i < predictions.size() / 2; i++) {
                curr_predictions[predictions[2*i]] += 
                    stod(predictions[2*i + 1]);
            }
        }
    }

    produce_output(curr_example, curr_predictions);

    return 0;
}