#ifndef CPP_RANDOMFUZZYTREE_H
#define CPP_RANDOMFUZZYTREE_H

#include "includes.h"
#include "math_functions.h"
#include "data.h"

struct Node {
    data_t data;
    vector<double> memberships;
    vector<pair<double, double>> ranges;
    function<double(item_t)> f;
    vector<Node *> children;
    Node *parent = nullptr;
    double entropy;
    double cardinality;
    int feature;
    map<string, double> weights;
    double cut_point;
    set<int> categorical_features_used;
    set<int> numerical_features_used;

    bool is_leaf() {
        return children.size() == 0;
    }
};

class RandomFuzzyTree {
public:
    Node root;
    int p;
    int feature_n;
    double a_cut;
    double min_gain_threshold;
    map<string, double> w;
    set<int> all_categorical_features;
    function<vector<int>(void)> random_feature_generator;
    bool is_rfg_set = false;
    int max_depth = -1;

    RandomFuzzyTree() {
    }

    void set_random_feature_generator(function<vector<int>(void)> rfg) {
        this->random_feature_generator = rfg;
        this->is_rfg_set = true;
    }

    void set_max_depth(const int &max_depth) {
        this->max_depth = max_depth;
    }

    map<string, double> &weights() {
        return w;
    }

    void fit(data_t &data,
             vector<range_t > &ranges,
             vector<int> categorical_features,
             vector<int> numerical_features,
             double a_cut = 0.5,
             double min_gain_threshold = 0.000001) {
        root = generate_root_node(data, ranges);
        this->a_cut = a_cut;
        this->feature_n = (int) ranges.size();
        this->p = int(ceil(sqrt(feature_n)));
        this->min_gain_threshold = min_gain_threshold;

        this->all_categorical_features = set<int>(categorical_features.begin(), categorical_features.end());

        for (auto &d : data) {
            w[d.second] = 1;
        }

        build_tree();
    }

    map<string, double> predict_memberships(item_t &x) {
        map<string, double> memberships_per_class;
        forward_pass(memberships_per_class, root, x, 1);

        return memberships_per_class;
    }

    void forward_pass(map<string, double> &memberships,
                      Node &node,
                      item_t &x,
                      double membership) {
        membership *= node.f(x);

        if (membership == 0) {
            return;
        }

        if (node.is_leaf()) {
            map<string, double> weights;
//            if (node.data.size() == 0) {
//                weights = root.weights;
//            } else {
                weights = node.weights;
//            }

            for (auto kv : weights) {
                memberships[kv.first] += kv.second * membership;
            }
        } else {
            for (int i = 0; i < node.children.size(); i++) {
                forward_pass(memberships,
                             *node.children[i],
                             x,
                             membership);
            }
        }
    }

    void build_tree() {
        queue<pair<Node *, int> > frontier;
        frontier.push(make_pair(&root, 0));

        while (frontier.size() != 0) {
            pair<Node *, int> curr = (pair<Node *, int> &&) frontier.front();
            Node *node = curr.first;
            int lvl = curr.second;

            if (this->max_depth == -1 or lvl <= this->max_depth) {
                if(node->data.size() != 0) {
                    vector<Node> children = get_best_children(node);
                    if (are_regular_children(children)) {
                        for (int i = 0; i < children.size(); i++) {
                            Node *child = new Node(children[i]);
                            node->children.push_back(child);

                            if (not(are_only_categorical() and
                                    no_categorical_left(node)) and
                                child->data.size() >= 2 and
                                not all_same(child) and
                                    (child->categorical_features_used.size() + child->numerical_features_used.size() < child->ranges.size())) {
                                frontier.push(make_pair(child, lvl + 1));
                            } else {
                                child->weights = weights(child);
                                child->data = data_t();
                                child->memberships = vector<double>();
                            }
                        }
                    }
                }
            }

            if(node->children.size() == 0) {
                node->weights = weights(node);
            }
            node->data = data_t();
            node->memberships = vector<double>();

            frontier.pop();
        }
    }

    bool all_same(Node *node) {
        for (int i = 1; i < node->data.size(); i++) {
            if (node->data[i].second.compare(node->data[0].second) != 0) {
                return false;
            }
        }

        return true;
    }

    bool no_categorical_left(Node *node) const {
        return node->categorical_features_used.size() == this->all_categorical_features.size();
    }

    bool are_only_categorical() const {
        return feature_n - all_categorical_features.size() == 0;
    }

    vector<Node> get_best_children(Node *node) {
        vector<int> features = generate_random_features(node);

        map<int, vector<Node>> feature_children;
        map<int, double> cut_points_per_feature;
        for (auto f : features) {
            double cut_point = -1;
            vector<Node> children = generate_feature_best_children(node, f, cut_point);
            cut_points_per_feature[f] = cut_point;

            if (are_regular_children(children)) {
                feature_children[f] = children;
            }
        }

        if (feature_children.size() == 0) {
            return vector<Node>();
        } else {
            vector<Node> best_children;
            double best_gain = -1000000;
            int best_feature;
            for (auto &kv : feature_children) {
                vector<Node> children = kv.second;
                double curr_gain = gain(children, node);
                if (curr_gain > best_gain) {
                    best_children = children;
                    best_gain = curr_gain;
                    best_feature = kv.first;
                }
            }

            node->feature = best_feature;
            node->cut_point = cut_points_per_feature[best_feature];

            python_plot(node, best_feature, cut_points_per_feature[best_feature]);

            return best_children;
        }
    }

    void python_plot(const Node *node, int f, double cut_point) const {
        return;

        string plot_arr_str = "python3 /home/faruk/workspace/thesis/reducer/plot.py \"[";
        int i = 0;
        for (auto &d : node->data) {
            string classification;
            if (d.second[d.second.size() - 1] == '\n') {
                classification = d.second.substr(0, d.second.length() - 1);
            } else {
                classification = d.second;
            }
            plot_arr_str +=
                    "(" + to_string(d.first[f]) + "," + to_string(node->memberships[i]) + "," + classification + "),";
            i++;
        }
        plot_arr_str += "]\" " + to_string(cut_point);
        system(plot_arr_str.c_str());
    }

    bool is_numerical_feature(int feature) {
        return !is_categorical_feature(feature);
    }

    bool is_categorical_feature(int feature) {
        return (all_categorical_features.find(feature) != all_categorical_features.end());
    }

    vector<Node> generate_best_children_categorical_feature(Node *pNode, int feature) {
        if (pNode->categorical_features_used.find(feature) == pNode->categorical_features_used.end()) {
            map<int, data_t> data_per_value;
            map<int, vector<double>> memberships_per_value;
            map<int, double> cardinality_per_value;

            for(int i = 0; i < pNode->data.size(); i++) {
                auto d = pNode->data[i];
                int key = (int) d.first[feature];
                data_per_value[key].push_back(d);
                memberships_per_value[key].push_back(pNode->memberships[i]);
                cardinality_per_value[key] += pNode->memberships[i];
            }

            vector<Node> children;

            for(auto kv : data_per_value) {
                auto categorical_features_used = pNode->categorical_features_used;
                categorical_features_used.insert(feature);

                Node child;
                child.data = kv.second;
                child.parent = pNode;
                child.ranges = pNode->ranges;
                child.memberships = memberships_per_value[kv.first];
                child.cardinality = cardinality_per_value[kv.first];
                child.entropy = fuzzy_entropy(&child);
//                child.weights = weights(&child);
                child.categorical_features_used = categorical_features_used;
                child.f = [feature, kv](pair<vector<double>, string> d) {
                    if ((int) d.first[feature] == kv.first) {
                        return 1.0;
                    } else {
                        return 0.0;
                    }
                };

                children.push_back(child);
            }

            return children;


            // old code
//            vector<Node> children;
//            for (int v = (int) pNode->ranges[feature].first; v <= pNode->ranges[feature].second; v++) {
//                data_t child_data;
//                vector<double> child_memberships;
//                for (int i = 0; i < pNode->data.size(); i++) {
//                    auto d = pNode->data[i];
//                    if ((int) d.first[feature] == (int) v) {
//                        child_data.push_back(d);
//                        child_memberships.push_back(pNode->memberships[i]);
//                    }
//                }
//
//                auto categorical_features_used = pNode->categorical_features_used;
//                categorical_features_used.insert(feature);
//
//                Node child;
//                child.data = child_data;
//                child.parent = pNode;
//                child.ranges = pNode->ranges;
//                child.memberships = child_memberships;
//                child.cardinality = fuzzy_cardinality(&child);
//                child.entropy = fuzzy_entropy(&child);
//                child.categorical_features_used = categorical_features_used;
//                child.f = [feature, v](pair<vector<double>, string> d) {
//                    if ((int) d.first[feature] == (int) v) {
//                        return 1.0;
//                    } else {
//                        return 0.0;
//                    }
//                };
//
//                children.push_back(child);
//            }
//
//            return children;
        } else {
            return vector<Node>();
        }
    }

    vector<Node> generate_feature_best_children(Node *node, int feature, double &cut_point) {
        if (is_numerical_feature(feature)) {
            return generate_best_children_numerical_feature(node, feature, cut_point);
        } else if (is_categorical_feature(feature)) {
            vector<Node> children = generate_best_children_categorical_feature(node, feature);
            return children;
        }
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"

    vector<Node> generate_discrete_children(Node *parent, int n, int feature) {
        double lower = parent->ranges[feature].first;
        double upper = parent->ranges[feature].second;

        double per_chunk = (upper - lower) / ( (n - 1) / 2 );

        vector<Node> children;
        vector<double> sums((unsigned long) n, 0.0);
        for(int i = 0; i < n; i++) {
            Node curr_child;

            curr_child.parent = parent;
            curr_child.ranges = root.ranges;
            curr_child.ranges[feature].first = (i - 1) * (per_chunk / 2);
            curr_child.ranges[feature].second = i * (per_chunk / 2);
            double point = lower + 0.5 * i * per_chunk;
            curr_child.f = triangular(point,
                                      per_chunk,
                                      feature);
            curr_child.cardinality = 0.0;

            children.push_back(curr_child);
        }

        for(int i = 0; i < parent->data.size(); i++) {
            auto &d = parent->data[i];
            int bucket = int((d.first[feature] - lower) / per_chunk);
            if(bucket >= children.size()) {
                bucket = (int) (children.size() - 1);
            }

            double curr_membership = parent->memberships[i] * children[bucket].f(d);
            children[bucket].memberships.push_back(curr_membership);
            children[bucket].cardinality += curr_membership;

            children[bucket + 1].memberships.push_back(curr_membership);
            children[bucket + 1].cardinality += curr_membership;
        }

        for(int i = 0; i < children.size(); i++) {
            children[i].entropy = fuzzy_entropy(&children[i]);
        }

        return children;
    }

    vector<Node> generate_best_children_numerical_feature(Node *node, int feature, double &cut_point) {
        string method = "RANDOM";
        int uniform_steps = 20;

        if (method == "UNIFORM" && node->data.size() < uniform_steps) {
            method = "FULL";
        }

        double lower = node->ranges[feature].first;
        double upper = node->ranges[feature].second;

        map<double, vector<Node>> children_per_point;

        if (method == "RANDOM") {
            int n = (int) ceil(log2(node->data.size()));

            for (int i = 0; i < n; i++) {
                std::random_device rd;  //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_int_distribution<> dis(0, (int) (node->data.size() - 1));

                int rand_ind = dis(gen);

                double p = node->data[rand_ind].first[feature];

                return generate_children_at_point(node, feature, p);

                children_per_point[p] = generate_children_at_point(node, feature, p);
            }
        } else if (method == "FULL") {
            set<double> points;
            for (auto &d : node->data) {
                points.insert(d.first[feature]);
            }

            double min_el = *min_element(points.begin(), points.end());
            double max_el = *max_element(points.begin(), points.end());
            pair<double, double> search_interval(min_el, max_el);
            double last_point = search_interval.first;

            for (double point : points) {
                if (point > search_interval.first and point < search_interval.second and !eq(point, lower) and
                    !eq(point, upper)) {
                    vector<Node> children =
                            generate_children_at_point(node, feature, point);
                    if (are_regular_children(children)) {
                        children_per_point[point] = children;
                    }
                    last_point = point;
                }
            }
        } else if (method == "UNIFORM") {
            double delta = (upper - lower) / uniform_steps;

            for (double point = lower; point <= upper; point += delta) {
                vector<Node> children =
                        generate_children_at_point(node, feature, point);
                if (are_regular_children(children)) {
                    children_per_point[point] = children;
                }
            }
        } else if(method == "DISCRETIZE") {
            vector<Node> best_children;
            double best_gain = -100000;
            int child_n = 5;
//            for(int child_n = 3; child_n < 10; child_n++) {
                vector<Node> children = generate_discrete_children(node, child_n, feature);
//            }

            return children;
        }

        if (children_per_point.size() == 0) {
            return vector<Node>();
        } else {
            vector<Node> best_children;
            double best_gain = -1000000;
            for (auto &kv : children_per_point) {
                vector<Node> children = kv.second;
                double curr_gain = gain(children, node);
                if (curr_gain > best_gain) {
                    best_children = children;
                    best_gain = curr_gain;
                    cut_point = kv.first;
                }
            }

            if (best_gain < min_gain_threshold) {
                return vector<Node>();
            }

            return best_children;
        }
    }

#pragma clang diagnostic pop

    double gain(vector<Node> &nodes, Node *parent) {
        double gain = parent->entropy;
        for (auto &n : nodes) {
            gain -= (n.cardinality / parent->cardinality) * n.entropy;
        }

        return gain;
    }

    bool are_regular_children(vector<Node> &children) {
        if (children.size() == 0) {
            return false;
        }

        int non_zero_n = 0;
        for (auto &child : children) {
            if (child.data.size() > 0) {
                non_zero_n++;
            }
        }
        return non_zero_n >= 2;
    }

    vector<Node> generate_children_at_point(Node *node, int feature, double point, int n = 3) {
        vector<Node> children;
        if (n == 3) {
            double lower = node->ranges[feature].first;
            double upper = node->ranges[feature].second;

            Node left_child;
            left_child.f = triangular(lower, 2 * (point - lower), feature);
            left_child.ranges = node->ranges;
            left_child.parent = node;
            left_child.ranges[feature].second = point;
            fill_node_properties(node, &left_child);
            children.push_back(left_child);

            Node middle_child;
            middle_child.f = composite_triangular(point,
                                                  2 * (point - lower),
                                                  2 * (upper - point),
                                                  feature);
            middle_child.ranges = node->ranges;
            middle_child.parent = node;
            fill_node_properties(node, &middle_child);
            children.push_back(middle_child);

            Node right_child;
            right_child.f = triangular(upper, 2 * (upper - point), feature);
            right_child.ranges = node->ranges;
            right_child.ranges[feature].first = point;
            right_child.parent = node;
            fill_node_properties(node, &right_child);
            children.push_back(right_child);

            return children;
        } else if (n == 5) {
            double lower = node->ranges[feature].first;
            double upper = node->ranges[feature].second;

            Node left_child;
            left_child.f = triangular(lower, (point - lower), feature);
            left_child.ranges = node->ranges;
            left_child.parent = node;
            left_child.ranges[feature].second = (lower + point) / 2;

            Node left_center_child;
            left_center_child.f = triangular((point + lower) / 2, (point - lower), feature);
            left_center_child.ranges = node->ranges;
            left_center_child.parent = node;
            left_center_child.ranges[feature].second = point;
            children.push_back(left_center_child);

            Node center_child;
            center_child.f = composite_triangular(point,
                                                  (point - lower),
                                                  (upper - point),
                                                  feature);
            center_child.ranges = node->ranges;
            center_child.parent = node;

            Node right_center_child;
            right_center_child.f = triangular((upper + point) / 2, (upper - point), feature);
            right_center_child.ranges = node->ranges;
            right_center_child.ranges[feature].first = point;
            right_center_child.parent = node;

            Node right_child;
            right_child.f = triangular(upper, (upper - point), feature);
            right_child.ranges = node->ranges;
            right_child.ranges[feature].first = (upper + point) / 2;
            right_child.parent = node;

            for(int i = 0; i < node->data.size(); i++) {
                pair<vector<double>, string> &d = node->data[i];
                double curr_membership = node->memberships[i];
                if(d.first[feature] < (lower + point) / 2) {
                    double left_f = left_child.f(d);
                    if(left_f >= a_cut) {
                        left_child.data.push_back(d);
                        left_child.memberships.push_back(left_f * curr_membership);
                        left_child.cardinality += left_f * curr_membership;
                    }

                    double left_center_f = left_center_child.f(d);
                    if(left_center_f >= a_cut) {
                        left_center_child.data.push_back(d);
                        left_center_child.memberships.push_back(left_center_f * curr_membership);
                        left_center_child.cardinality += left_center_f * curr_membership;
                    }
                } else if(d.first[feature] < point) {
                    double left_center_f = left_center_child.f(d);
                    if(left_center_f >= a_cut) {
                        left_center_child.data.push_back(d);
                        left_center_child.memberships.push_back(left_center_f * curr_membership);
                        left_center_child.cardinality += left_center_f * curr_membership;
                    }

                    double center_f = center_child.f(d);
                    if(center_f >= a_cut) {
                        center_child.data.push_back(d);
                        center_child.memberships.push_back(center_f * curr_membership);
                        center_child.cardinality += center_f * curr_membership;
                    }
                } else if(d.first[feature] < (point + upper) / 2) {
                    double center_f = center_child.f(d);
                    if(center_f >= a_cut) {
                        center_child.data.push_back(d);
                        center_child.memberships.push_back(center_f * curr_membership);
                        center_child.cardinality += center_f * curr_membership;
                    }

                    double right_center_f = right_center_child.f(d);
                    if(right_center_f >= a_cut) {
                        right_center_child.data.push_back(d);
                        right_center_child.memberships.push_back(right_center_f * curr_membership);
                        right_center_child.cardinality += right_center_f * curr_membership;
                    }
                } else {
                    double right_center_f = right_center_child.f(d);
                    if(right_center_f >= a_cut) {
                        right_center_child.data.push_back(d);
                        right_center_child.memberships.push_back(right_center_f * curr_membership);
                        right_center_child.cardinality += right_center_f * curr_membership;
                    }

                    double right_f = right_child.f(d);
                    if(right_f >= a_cut) {
                        right_child.data.push_back(d);
                        right_child.memberships.push_back(right_f * curr_membership);
                        right_child.cardinality += right_f * curr_membership;
                    }
                }
            }

            children.push_back(left_child);
            children.push_back(left_center_child);
            children.push_back(center_child);
            children.push_back(right_center_child);
            children.push_back(right_child);

            return children;
        } else if (n == 2) {
            double lower = node->ranges[feature].first;
            double upper = node->ranges[feature].second;

            double flat_percentage = 0.8;
            double left_mid = lower + flat_percentage * (point - lower);
            double right_mid = point + (1 - flat_percentage) * (upper - point);

            Node left_child;
            left_child.f = trapezoid_left(left_mid,
                                          (left_mid - lower),
                                          (upper - left_mid),
                                          feature);
            left_child.ranges = node->ranges;
            left_child.parent = node;
            left_child.ranges[feature].second = right_mid;
            left_child.numerical_features_used.insert(feature);

            Node middle_child;
            middle_child.f = composite_triangular(point,
                                                  (point - left_mid),
                                                  (right_mid - point),
                                                  feature);
            middle_child.ranges = node->ranges;
            middle_child.parent = node;
            middle_child.ranges[feature].first = left_mid;
            middle_child.ranges[feature].second = right_mid;
            middle_child.numerical_features_used.insert(feature);

            Node right_child;
            right_child.f = trapezoid_right(right_mid,
                                            (right_mid - lower),
                                            (upper - point),
                                            feature);
            right_child.ranges = node->ranges;
            right_child.ranges[feature].first = left_mid;
            right_child.parent = node;
            right_child.numerical_features_used.insert(feature);

            fill_node_properties_modified(node, &left_child, &middle_child, &right_child, left_mid, right_mid, feature, point);

            children.push_back(left_child);
            children.push_back(right_child);
//            children.push_back(middle_child);

            return children;
        }
    }

    void fill_node_properties_modified(Node *parent,
                                       Node *left,
                                       Node *middle,
                                       Node *right,
                                       double left_mid,
                                       double right_mid,
                                       int feature,
                                       double point) {
        double left_sum = 0;
        double right_sum = 0;
        double middle_sum = 0;

        for(int i = 0; i < parent->data.size(); i++) {
            auto d = parent->data[i];

            if(left->f(d) > 0.5) {
                left->data.push_back(d);
                double m = parent->memberships[i] * left->f(d);
                left->memberships.push_back(m);
                left_sum += m;
            }

            if(right->f(d) > 0.5) {
                right->data.push_back(d);
                double m = parent->memberships[i] * right->f(d);
                right->memberships.push_back(m);
                right_sum += m;
            }

            // jello
            continue;

            if(d.first[feature] < left_mid) {
                left->data.push_back(d);
                left->memberships.push_back(parent->memberships[i]);
                left_sum += parent->memberships[i];
            } else if(d.first[feature] < point) {
                double curr_membership = parent->memberships[i] * left->f(d);
                if(left->f(d) > a_cut) {
                    left->data.push_back(d);
                    left->memberships.push_back(curr_membership);
                    left_sum += curr_membership;
                }

                curr_membership = parent->memberships[i] * middle->f(d);
                if(middle->f(d) > a_cut) {
                    middle->data.push_back(d);
                    middle->memberships.push_back(curr_membership);
                    middle_sum += curr_membership;
                }
            } else if(d.first[feature] < right_mid) {
                double curr_membership = parent->memberships[i] * right->f(d);
                if(right->f(d) > a_cut) {
                    right->data.push_back(d);
                    right->memberships.push_back(curr_membership);
                    right_sum += curr_membership;
                }

                curr_membership = parent->memberships[i] * middle->f(d);
                if(middle->f(d) > a_cut) {
                    middle->data.push_back(d);
                    middle->memberships.push_back(curr_membership);
                    middle_sum += curr_membership;
                }
            } else {
                right->data.push_back(d);
                right->memberships.push_back(parent->memberships[i]);
                right_sum += parent->memberships[i];
            }
        }

        left->cardinality = left_sum;
        right->cardinality = right_sum;
//        middle->cardinality = middle_sum;

        left->entropy = fuzzy_entropy(left);
        right->entropy = fuzzy_entropy(right);
//        middle->entropy = fuzzy_entropy(middle);
    }

    void fill_node_properties(Node *parent, Node *node) {
        if (parent != node) {
            node->data = parent->data;
            node->memberships = parent->memberships;
        }

        vector<double> local_memberships = get_local_memberships(node);
        node->memberships = vec_mult(node->memberships, local_memberships);

        vector<item_t > next_data;
        vector<double> next_memberships;

        for (int i = 0; i < node->memberships.size(); i++) {
            if (local_memberships[i] > a_cut) {
                next_data.push_back(node->data[i]);
                next_memberships.push_back(node->memberships[i]);
            }
        }

        if (node != parent) {
            node->data = next_data;
            node->memberships = next_memberships;
        }

        node->cardinality = fuzzy_cardinality(node);
        node->entropy = fuzzy_entropy(node);
    }

    map<string, double> weights(Node *node) {
        if (node->data.size() != 0) {
            map<string, double> weights_values = {};

            double total = 0;
            for (int i = 0; i < node->data.size(); i++) {
                double val = node->memberships[i];
                weights_values[node->data[i].second] += val;
                total += val;
            }

            for (auto kv : weights_values) {
                weights_values[kv.first] /= total;
            }

            return weights_values;
        } else {
            return map<string, double>();
        }
    }

    double fuzzy_cardinality(Node *node) {
        double cardinality = 0;
        for (auto &val : node->memberships) {
            cardinality += val;
        }

        return cardinality;
    }

    double fuzzy_entropy(Node *node) {
        if(node->data.size() == 0) {
            return 0;
        }

        map<string, double> memberships_per_class;
        for (int i = 0; i < node->data.size(); i++) {
            string cls = node->data[i].second;
            memberships_per_class[cls] += node->memberships[i];
        }

        double entropy = 0;

        vector<double> probs;
        double sum = 0;
        for (auto kv : memberships_per_class) {
            double proba = kv.second / node->cardinality;
            double softmax_val = proba;//exp(proba);
            probs.push_back(softmax_val);
            sum += softmax_val;
        }

        for(double proba : probs) {
            proba /= sum;
            if (proba != 0) {
                entropy -= proba * log2(proba);
            }
        }

        return entropy;
    }

    vector<double> vec_mult(vector<double> &a, vector<double> &b) {
        vector<double> res;
        for (int i = 0; i < a.size(); i++) {
            res.push_back(a[i] * b[i]);
        }

        return res;
    }

    vector<double> get_local_memberships(Node *node) {
        vector<double> local_memberships;

        for (auto &d : node->data) {
            double curr_memb = node->f(d);
            local_memberships.push_back(curr_memb);
        }

        return local_memberships;
    }

    vector<int> generate_random_features(Node *node) {
        if (this->is_rfg_set) {
            return this->random_feature_generator();
        }

        int req = int(ceil(log2(node->ranges.size() - ( node->numerical_features_used.size() + node->categorical_features_used.size() ))));

        vector<int> features;
        for (int i = 0; features.size() < req; i++) {
            int n = feature_n;

            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<> dis(0, (int) (n - 1));

            int rand_ind = dis(gen);

//            and not in(node->numerical_features_used, rand_ind)
            if (not in(node->categorical_features_used, rand_ind)) {
                features.push_back(rand_ind);
            }
        }

        return features;
    }

    bool in(const set<int> &s, int el) {
        return s.find(el) != s.end();
    }

    bool chosen(vector<int> &features, int rand_ind) const {
        return find(features.begin(), features.end(), rand_ind) != features.end();
    }

    Node generate_root_node(data_t &data, vector<range_t > &ranges) {
        Node root = Node();
        root.data = data;
        root.ranges = ranges;
        root.memberships = vector<double>(data.size(), 1.0);
        root.f = function<double(item_t)>([](item_t i) { return 1.0; });

        auto prev_data = root.data;

        fill_node_properties(&root, &root);

        return root;
    }
};

#endif //CPP_RANDOMFUZZYTREE_H
