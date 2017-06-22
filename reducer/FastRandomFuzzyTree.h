//
// Created by faruk on 6/22/17.
//

#ifndef CPP_FASTRANDOMFUZZYTREE_H
#define CPP_FASTRANDOMFUZZYTREE_H

#define ul unsigned long

#include <climits>
#include "includes.h"
#include "math_functions.h"
#include "data.h"

class FastRandomFuzzyTree {
public:
    struct Cut {
        double point;
        int feature;
        double gain;
    };

    enum Side {
        LEFT, RIGHT
    };

    struct Node {
        ul from, to;
        vector<double> membs;
        map<int, pair<double, double>> changed_ranges;
        vector<ul> children;
        Side side;
        int feature;
        double entropy;

        Node() {}

        void clear() {
            vector<double>().swap(membs);
            map<int, pair<double, double>>().swap(changed_ranges);
        }
    };

    double a_cut;
    double min_gain;
    double q;
    data_t data;
    ul feature_n;
    set<int> categorical_features;
    int verbose;
    vector<range_t > ranges;
    vector<Node> nodes;

    FastRandomFuzzyTree() {}

    void set_verbose(int verbose) {
        this->verbose = verbose;
    }

    void fit(data_t &data,
             vector<range_t > &ranges,
             vector<int> categorical_features,
             double a_cut = 0.5,
             double min_gain = 0.000001,
             double q = 0.2) {
        this->categorical_features = set<int>(categorical_features.begin(),
                                              categorical_features.end());
        this->a_cut = a_cut;
        this->ranges = ranges;
        this->min_gain = min_gain;
        this->data = data;
        this->feature_n = ranges.size();
        this->verbose = verbose;
        this->q = q;

        // to avoid a lot of allocations
        nodes.reserve(data.size());

        vector<int> all_features(ranges.size());
        for (int i = 0; i < all_features.size(); i++) {
            all_features[i] = i;
        }

        Node root = Node();
        root.from = 0;
        root.to = data.size();
        root.membs = vector<double>(data.size(), 1.0);

        nodes.push_back(root);

        build_children(0, all_features);
    }

    bool build_children(int parent, vector<int> features_left) {
        // create empty children, will be filled
        nodes.push_back(Node());
        nodes.push_back(Node());

        // assign children to parent
        nodes[parent].children = vector<ul>(2);
        nodes[parent].children[0] = nodes.size() - 2;
        nodes[parent].children[1] = nodes.size() - 1;

        ul from = nodes[parent].from;
        ul to = nodes[parent].to;

        random_shuffle(features_left.begin(),
                       features_left.end());

        Cut best_cut;
        best_cut.gain = INT_MIN;

        for (int i = 0;
             i < sqrt(features_left.size()) && i < features_left.size();
             i++) {

            int feature = features_left[i];

            sort(data.begin() + from,
                 data.begin() + to,
                 [feature](item_t a, item_t b) -> bool {
                     return a.first[feature] < b.first[feature];
                 });

            if (verbose == 2) {
                cout << "Current slice: ";
                for (auto it = data.begin() + from; it != data.begin() + to; it++) {
                    cout << it->first[feature] << ", ";
                }
                cout << endl;
            }

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<ul> uni(from + 1, to - 2);

            ul random_index = uni(rng);
            double cut_point = data[random_index].first[feature];

            if (verbose == 2) {
                cout << "Cut point: " << cut_point << endl;
            }

            update_node(parent,
                        feature,
                        cut_point);


        }
    }

    double update_node(int parent,
                       int feature,
                       double point) {
        // assumes [from, to] range is sorted

        Node &parent_node = nodes[parent];

        ul from = parent_node.from;
        ul to = parent_node.to;

        // left child
        ul &left_child_ind = parent_node.children[0];
        Node &left_child = nodes[left_child_ind];
        left_child.side = LEFT;
        left_child.feature = feature;

        if (parent_node.changed_ranges.count(feature) == 0) {
            left_child.changed_ranges[feature] = ranges[feature];
        } else {
            left_child.changed_ranges[feature] = parent_node.changed_ranges[feature];
        }
        left_child.changed_ranges[feature].second = point;

        ul i = from;
        // exact point always goes to the right child
        // it is not important as long as choice is
        // consistent
        map<string, double> memb_per_class;
        while (data[i].first[feature] <= point) {
            double memb = membership(left_child_ind,
                                     data[i].first[feature]);
            left_child.membs.push_back(memb);
            memb_per_class[data[i].second] += memb;
            i++;
        }
        update_entropy(left_child_ind, memb_per_class);

        left_child.from = from;
        left_child.to = i;

        // right child
    }

    void update_entropy(ul node, const map<string, double> &membs) {
        nodes[node].entropy = 0;

        double total = 0;
        for (auto &kv : membs) {
            total += kv.second;
        }

        for (auto &kv : membs) {
            double p = kv.second / total;
            nodes[node].entropy -= p * log2(p);
        }
    }

    double membership(ul node, double val) {
        int feature = nodes[node].feature;
        double a = nodes[node].changed_ranges[feature].first;
        double b = nodes[node].changed_ranges[feature].second;

        double memb = 0;
        if (nodes[node].side == LEFT) {
            memb = left_trapezoidal(a, b, val, q);
        } else if (nodes[node].side == RIGHT) {
            memb = right_trapezoidal(a, b, val, q);
        }

        return memb;
    }

    static double left_trapezoidal(double a,
                                   double b,
                                   double val,
                                   double q) {
        if (val <= a || val > b) {
            return 0;
        } else if (val <= b - q * (b - a)) {
            return 1;
        } else {
            double k = 1 / (b - a);
            return 1 - k * (val - a);
        }
    }

    static double right_trapezoidal(double a,
                                    double b,
                                    double val,
                                    double q) {
        return left_trapezoidal(b, b + (b - a), b + (val - a), q);
    }

};


#endif //CPP_FASTRANDOMFUZZYTREE_H
