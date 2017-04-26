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

#define SPECIAL_VALUE 225883.225883
#define PRINT_VALUE 225884.225883
#define PRINT_NAME 235884.225883

function<double(pair<vector<double>, string>)> triangular(double center, double width, int feature, double name = SPECIAL_VALUE + 1) {
    return ([center, width, name, feature](pair<vector<double>, string> item) {
        double x = item.first[feature];
        double r = width / 2;
        double k = 1 / r;

        double left = center - r;
        double right = center + r;

        if(eq(x, SPECIAL_VALUE)) {
            return name;
        } else if(eq(x, PRINT_VALUE)) {
//            cout << left << " to " << right << " k = " << k << " center: " << center;
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
    });
}

function<double(pair<vector<double>, string>)> composite_triangular(double center, double left_width, double right_width, int feature, double name = SPECIAL_VALUE + 1) {
    auto left_f = triangular(center, left_width, feature, name);
    auto right_f = triangular(center, right_width, feature, name);

    return [center, left_f, right_f, feature](pair<vector<double>, string> item) {
        double x = item.first[feature];

        if(x < center) {
            return left_f(item);
        } else {
            return right_f(item);
        }
    };
}