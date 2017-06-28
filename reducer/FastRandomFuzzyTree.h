//
// Created by faruk on 6/22/17.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#ifndef CPP_FASTRANDOMFUZZYTREE_H
#define CPP_FASTRANDOMFUZZYTREE_H

#define ul unsigned long
#define EPS 1e-7

#include <climits>
#include <stack>
#include "includes.h"
#include "math_functions.h"
#include "data.h"
#include <unordered_map>

class FastRandomFuzzyTree {
public:
    struct Split {
        double point;
        int feature;
        double gain;
        ul index;

        static Split get_worst_split() {
            Split split;
            split.gain = INT_MIN;

            return split;
        }

        bool operator>(Split &s) {
            return gain > s.gain;
        }
    };

    enum Side {
        LEFT, RIGHT
    };

    struct Node {
        ul from, to;
        string label;

        ul parent_ind;
        vector<ul> children;

        Side side;
        int feature;
        double point;

        bool _is_root = false;

        unordered_map<string, double> per_class;

        Node() {}

        bool is_root() {
            return _is_root;
        }

        bool is_leaf() {
            return children.size() == 0;
        }
    };

    double a_cut;
    double min_gain;
    double q;
    data_t *data;
    ul feature_n;
    set<int> categorical_features;
    int verbose;
    vector<range_t > ranges;
    vector<ul> data_inds;
    vector<Node> nodes;
    double min_elements;
    vector<int> features_left;

    FastRandomFuzzyTree() {}

    void set_verbose(int verbose) {
        this->verbose = verbose;
    }

    map<string, double> predict_memberships(item_t &x) {
        map<string, double> membs;
        forward_pass(x, 0, membs);

        return membs;
    }

    void forward_pass(item_t &x,
                      ul node_ind,
                      map<string, double> &membs) {
        if (nodes[node_ind].is_leaf()) {
            for(auto &kv : nodes[node_ind].per_class) {
                membs[kv.first] = kv.second;
            }
        } else {
            double val = x.first[nodes[node_ind].feature];
            if (val < nodes[node_ind].point) {
                forward_pass(x, nodes[node_ind].children[0], membs);
            } else {
                forward_pass(x, nodes[node_ind].children[1], membs);
            }
        }
    }

    void fit(data_t &data,
             vector<range_t > &ranges,
             vector<int> categorical_features,
             double a_cut = 0.5,
             double min_gain = 1e-7,
             double q = 0.2,
             double min_elements = 2) {

        this->categorical_features = set<int>(categorical_features.begin(),
                                              categorical_features.end());
        this->a_cut = a_cut;
        this->ranges = ranges;
        this->min_gain = min_gain;
        this->data = &data;
        this->feature_n = ranges.size();
        this->verbose = verbose;
        this->q = q;
        this->min_elements = min_elements;

        this->data_inds = vector<ul>(this->data->size());
        for(ul i = 0; i < this->data->size(); i++) {
            this->data_inds[i] = i;
        }

        build_tree();

//        print();
    }

    void build_tree() {
        // to avoid a lot of allocations
        nodes.reserve((ul) (1.5 * data->size()));

        build_root();

        // bfs tree build
        // dfs takes up too much memory
        queue<ul> frontier;
        frontier.push(0);
        while (frontier.size() != 0) {
            ul curr = frontier.front();
            frontier.pop();

            if (build_children(curr)) {
                for (auto &c : nodes[curr].children) {
                    frontier.push(c);
                }
            } else {
                build_leaf(curr);
            }
        }
    }

    void print(ul node_ind = 0, int lvl = 0) {
        Node &node = nodes[node_ind];
        if(node.is_leaf()) {
            cout << string(lvl, '\t') << "(";
            for(ul i = node.from; i < node.to; i++) {
                cout << data->operator[](data_inds[i]).second << ", ";
            }
            cout << ") -> " << nodes[node_ind].label << endl;
        } else {
            cout << string(lvl, '\t') << "(" << node.from <<
                 ", " << node.to << ")" << endl;
        }

        for (auto &c : node.children) {
            print(c, lvl + 1);
        }
    }

    void build_root() {
        vector<int> all_features(ranges.size());
        for (int i = 0; i < all_features.size(); i++) {
            all_features[i] = i;
        }

        this->features_left = all_features;

        Node root = Node();
        root._is_root = true;
        root.from = 0;
        root.to = data->size();
        nodes.push_back(root);
    }

    void build_leaf(ul node_ind) {
        unsigned long from = nodes[node_ind].from;
        unsigned long to = nodes[node_ind].to;

        int max_cnt = -1000;

        unordered_map<string, double> per_class;
        double total = 0;
        for(ul i = from; i < to; i++) {
            string &curr_label = data->operator[](data_inds[i]).second;
            per_class[curr_label]++;
            total++;
        }

        double max_perc = INT_MIN;
        string max_label;
        for(auto &kv : per_class) {
            if(kv.second > max_perc) {
                max_perc = kv.second;
                max_label = kv.first;
            }
            kv.second /= total;
        }

        nodes[node_ind].per_class = per_class;
        nodes[node_ind].label = max_label;
    }

    bool build_children(ul parent_ind) {
        if (nodes[parent_ind].to - nodes[parent_ind].from < min_elements) {
            return false;
        }

        ul from = nodes[parent_ind].from;
        ul to = nodes[parent_ind].to;

        random_shuffle(features_left.begin(),
                       features_left.end());


        Split best_split = Split::get_worst_split();
        for (int i = 0;
             i < sqrt(features_left.size());
             i++) {

            int feature = features_left[i];
            Split split = get_split(feature, from, to);
            if (split > best_split) {
                best_split = split;
            }
        }

        if (best_split.gain < min_gain) {
            return false;
        } else {
            apply_split(from, to, parent_ind, best_split);
            return true;
        }
    }

    bool all_same(ul node_ind) {
        if (nodes[node_ind].is_root()) {
            return false;
        }

        unsigned long from = nodes[node_ind].from;
        unsigned long to = nodes[node_ind].to;
        for (ul i = from; i < to; i++) {
            if (!eq(data->operator[](data_inds[i]).first[nodes[node_ind].feature], data->operator[](data_inds[from]).first[nodes[node_ind].feature])) {
                return false;
            }
        }

        return true;
    }

    void apply_split(ul from, ul to, ul node_ind, Split split) {
        nodes[node_ind].point = split.point;

        Node left_child, right_child;
        left_child.feature = split.feature;
        right_child.feature = split.feature;

        left_child.from = from;
        right_child.to = to;

        left_child.to = split.index;
        right_child.from = split.index;

        nodes.push_back(left_child);
        nodes[node_ind].children.push_back(nodes.size() - 1);

        nodes.push_back(right_child);
        nodes[node_ind].children.push_back(nodes.size() - 1);
    }

    Split get_split(int feature, ul from, ul to) {
        sort(data_inds.begin() + from,
             data_inds.begin() + to,
             [feature, this](ul a, ul b) -> bool {
                 return this->data->operator[](a).first[feature] < this->data->operator[](b).first[feature];
             });

        vector< map<string, int> > histograms;
        map<string, int> curr_histogram;
        for(ul i = from; i < to; i++) {
            curr_histogram[ this->data->operator[](data_inds[i]).second ]++;
            histograms.push_back(curr_histogram);
        }

        Split best_split = Split::get_worst_split();

        for (int j = 0; j < sqrt(to - from); j++) {
            ul random_index = (unsigned long) (from +  (rand() / (1.0 * RAND_MAX)) * (to - from - 1));
            double point = data->operator[](data_inds[random_index]).first[feature];

            Split split;
            split.feature = feature;
            split.point = point;
            split.index = random_index;
            split.gain = calc_gain(feature, from, to, random_index, histograms);

            if (split > best_split) {
                best_split = split;
            }
        }

        return best_split;
    }

    struct NodeProperties {
        int cardinality;
        double entropy;
    };

    double calc_gain(int feature, ul from, ul to, ul index, vector< map<string, int> > &histograms) {
        NodeProperties left_child_prop = calc_properties(feature, from, to, index, LEFT);
        NodeProperties right_child_prop = calc_properties(feature, from, to, index, RIGHT);

        double entropy = 0;
        double ttl = 0;
        unordered_map<string, int> per_class;
        for (ul i = from; i < to; i++) {
            per_class[data->operator[](data_inds[i]).second]++;
            ttl++;
        }
        for (auto &kv : per_class) {
            double p = kv.second / ttl;
            entropy += p * (1 - p);
        }

        double total = to - from;
        double left_perc = left_child_prop.cardinality / total;
        double right_perc = right_child_prop.cardinality / total;
        double gain = left_perc * left_child_prop.entropy +
                      right_perc * right_child_prop.entropy;
        gain = entropy - gain;
        return gain;
    }

    NodeProperties calc_properties(int feature, ul from, ul to, ul index, Side side) {
        ul a = 0, b = 0;
        if (side == LEFT) {
            a = from;
            b = index;
        } else if (side == RIGHT) {
            a = index;
            b = to;
        }

        NodeProperties node_properties;
        unordered_map<string, double> per_class;
        double total = 0;
        for (ul i = a; i < b; i++) {
            pair<vector<double>, string> &x = data->operator[](data_inds[i]);
            per_class[x.second]++;
            total++;
        }
        double entropy = calc_entropy(per_class, total);
        node_properties.entropy = entropy;
        node_properties.cardinality = (int) total;

        return node_properties;
    }

    double calc_entropy(unordered_map<string, double> &per_class, double total) {
        double entropy = 0;

        for (auto &kv : per_class) {
            double p = kv.second / total;
            entropy += p * (1 - p);
        }

        return entropy;
    }

    double membership(ul node_ind, double val) {
        if (nodes[node_ind]._is_root) {
            return 1.0;
        }

        int feature = nodes[node_ind].feature;

        return 0; // TODO: change !!!!
    }

    static double left_trapezoidal(double a,
                                   double b,
                                   double val,
                                   double q) {
        if (val < a || val > b) {
            return 0;
        } else if (val <= b - q * (b - a)) {
            return 1;
        } else {
            double peak = b - q * (b - a);
            double k = 1 / (b - peak);
            return 1 - k * (val - peak);
        }
    }

    static double right_trapezoidal(double a,
                                    double b,
                                    double val,
                                    double q) {
        if (val < a or val > b) {
            return 0;
        } else if (val > a + q * (b - a)) {
            return 1;
        } else {
            double peak = a + q * (b - a);
            double k = 1.0 / (peak - a);
            return k * (val - a);
        }
    }

};


#endif //CPP_FASTRANDOMFUZZYTREE_H

#pragma clang diagnostic pop