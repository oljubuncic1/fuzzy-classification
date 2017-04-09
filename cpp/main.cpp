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
#include <atomic>
#include <queue>
#include <thread>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "data.h"
#include "math_functions.h"

using namespace std;

#define item_t pair<vector<double>, string>
#define data_t vector<item_t>

struct Node {
    data_t data;
    vector<double> memberships;
    vector<pair<double, double>> ranges;
    function<double(item_t)> f;
    vector<Node*> children;
    double entropy;
    double cardinality;
    map<string, double> weights;

    bool is_leaf() {
        return children.size() == 0;
    }
};

class RandomFuzzyTree {
private:
    Node root;
    int max_depth;
    int p;
    int feature_n;
    double a_cut;
    double min_gain_threshold;
    map<string, double> w;
public:
    RandomFuzzyTree() {

    }

    map<string, double>& weights() {
        return w;
    }

    void fit(data_t &data,
             vector<range_t> &ranges,
             int max_depth=10000,
             int p=-1,
             double a_cut=0.5,
             double min_gain_threshold=0.000001) {
        root = generate_root_node(data, ranges);
        this->max_depth = max_depth;
        this->a_cut = a_cut;
        this->feature_n = ranges.size();
        this->p = int( sqrt(feature_n) );
        this->min_gain_threshold = min_gain_threshold;

        for(auto &d : data) {
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

        if( node.is_leaf() ) {
            for(auto kv : node.weights) {
                memberships[kv.first] += kv.second * membership;
            }
        } else {
            for(int i = 0; i < node.children.size(); i++) {
                forward_pass(memberships,
                             *node.children[i],
                             x,
                             membership);
            }
        }
    }

    void build_tree() {
        queue< pair<Node*, int> > frontier;
        frontier.push(make_pair(&root, 0));

        while(frontier.size() != 0) {
            pair<Node*, int> curr = (pair<Node *, int> &&) frontier.front();
            Node *node = curr.first;
            int lvl = curr.second;

            vector<Node> children = get_best_children(node);
            if(children.size() != 0) { // and lvl < max_depth) {
                for(int i = 0; i < children.size(); i++) {
                    Node *child = new Node(children[i]);
                    frontier.push(make_pair(child, lvl + 1));
                    node->children.push_back(child);
                }
            }

            frontier.pop();
        }
    }

    vector<Node> get_best_children(Node *node) {
        vector<int> features = generate_random_features();
        map<int, vector<Node>> feature_children;
        for(auto f : features) {
            vector<Node> children = generate_feature_best_children(node, f);
            if(are_regular_children(children)) {
                feature_children[f] = children;
            }
        }

        if(feature_children.size() == 0) {
            return vector<Node>();
        } else {
            vector<Node> best_children;
            double best_gain = -1000000;
            for(auto &kv : feature_children) {
                vector<Node> children = kv.second;
                double curr_gain = gain(children, node);
                if(curr_gain > best_gain) {
                    best_children = children;
                    best_gain = curr_gain;
                }
            }

            return best_children;
        }
    }

    vector<Node> generate_feature_best_children(Node *node, int feature) {
        set<double> points;
        for(auto &d : node->data) {
            points.insert(d.first[feature]);
        }

        double lower = node->ranges[feature].first;
        double upper = node->ranges[feature].second;


        double min_el = *min_element(points.begin(), points.end());
        double max_el = *max_element(points.begin(), points.end());
        pair<double, double> search_interval(min_el, max_el);
        double interval_len_threshold = (max_el - min_el) / 16;
        while(search_interval.second - search_interval.first > interval_len_threshold) {
            double a = search_interval.first;
            double b = search_interval.second;

            double mid = (a + b) / 2;

            double eps = 0.005; //(b - a) / 20;

            double left_x = mid - eps;
            vector<Node> left_children =
                    generate_children_at_point(node, feature, left_x);
            double left_y = gain(left_children, node);

            double right_x = mid + eps;
            vector<Node> right_children =
                    generate_children_at_point(node, feature, right_x);
            double right_y = gain(right_children, node);

            if(left_y < right_y) {
                search_interval.first = mid;
            } else {
                search_interval.second = mid;
            }
        }

        map<double, vector<Node>> children_per_point;
        for(double point : points) {
            if(point > search_interval.first and point < search_interval.second and !eq(point, lower) and !eq(point, upper)) {
                vector<Node> children =
                        generate_children_at_point(node, feature, point);
                if( are_regular_children(children) ) {
                    children_per_point[point] = children;
                }
            }
        }

        if(children_per_point.size() == 0) {
            return vector<Node>();
        } else {
            vector<Node> best_children;
            double best_gain = -1000000;
            for(auto &kv : children_per_point) {
                vector<Node> children = kv.second;
                double curr_gain = gain(children, node);
                if(curr_gain > best_gain) {
                    best_children = children;
                    best_gain = curr_gain;
                }
            }

            if(best_gain < min_gain_threshold) {
                return vector<Node>();
            }

            return best_children;
        }
    }

    double gain(vector<Node> &nodes, Node *parent) {
        double gain = parent->entropy;
        for(auto &n : nodes) {
            gain -= (n.cardinality / parent->cardinality) * n.entropy;
        }

        return gain;
    }

    bool are_regular_children(vector<Node> &children) {
        int non_zero_n = 0;
        for(auto &child : children) {
            if(child.data.size() != 0) {
                non_zero_n++;
            }
        }
        return non_zero_n >= 2;
    }

    vector<Node> generate_children_at_point(Node *node, int feature, double point) {
        vector<Node> children;

        double lower = node->ranges[feature].first;
        double upper = node->ranges[feature].second;

        Node left_child;
        left_child.f = triangular(lower, 2 * (point - lower), feature);
        left_child.ranges = node->ranges;
        left_child.ranges[feature].second = point;
        fill_node_properties(node, &left_child);
        children.push_back(left_child);

        Node middle_child;
        middle_child.f = composite_triangular(point,
                                              2 * (point - lower),
                                              2 * (upper - point),
                                              feature);
        middle_child.ranges = node->ranges;
        fill_node_properties(node, &middle_child);
        children.push_back(middle_child);

        Node right_child;
        right_child.f = triangular(upper, 2 * (upper - point), feature);
        right_child.ranges = node->ranges;
        right_child.ranges[feature].first = point;
        fill_node_properties(node, &right_child);
        children.push_back(right_child);

        return children;
    }

    void fill_node_properties(Node *parent, Node *node) {
        node->data = parent->data;

        node->memberships = parent->memberships;
        vector<double> local_memberships = get_local_memberships(node);
        node->memberships = vec_mult(node->memberships, local_memberships);

        vector<item_t> next_data;
        vector<double> next_memberships;

        for(int i = 0; i < node->memberships.size(); i++) {
            if(local_memberships[i] > a_cut) {
                next_data.push_back(node->data[i]);
                next_memberships.push_back(node->memberships[i] *
                                           node->f(node->data[i]));
            }
        }

        node->data = next_data;
        node->memberships = next_memberships;
        node->cardinality = fuzzy_cardinality(node);
        node->entropy = fuzzy_entropy(node);
        node->weights = weights(node, parent);
    }

    map<string, double> weights(Node *node, Node *parent=nullptr) {
        if(node->data.size() != 0) {
            map<string, double> weights_values = {};

            double total = 0;
            for(int i = 0; i < node->data.size(); i++) {
                double val = node->memberships[i];
                weights_values[node->data[i].second] += val;
                total +=  val;
            }

            for(auto kv : weights_values) {
                weights_values[kv.first] /= total;
            }

            return weights_values;
        } else {
            return map<string, double>();
            // return weights(parent);
        }
    }

    double fuzzy_cardinality(Node *node) {
        double cardinality = 0;
        for(auto &val : node->memberships) {
            cardinality += val;
        }

        return cardinality;
    }

    double fuzzy_entropy(Node *node) {
        map<string, double> memberships_per_class;
        for(int i = 0; i < node->data.size(); i++) {
            string cls = node->data[i].second;
            memberships_per_class[cls] += node->memberships[i];
        }

        double entropy = 0;
        for(auto kv : memberships_per_class) {
            double proba = kv.second / node->cardinality;
            if(proba != 0) {
                entropy -= proba * log2(proba);
            }
        }

        return entropy;
    }

    vector<double> vec_mult(vector<double> &a, vector<double> &b) {
        vector<double> res;
        for(int i = 0; i < a.size(); i++) {
            res.push_back(a[i] * b[i]);
        }

        return res;
    }

    vector<double> get_local_memberships(Node *node) {
        vector<double> local_memberships;

        for(auto &d : node->data) {
            double curr_memb = node->f(d);
            local_memberships.push_back(curr_memb);
        }

        return local_memberships;
    }

    vector<int> generate_random_features() {
        vector<int> features;
        for(int i = 0; i < p; i++) {
            int rand_ind = rand();
            int n = feature_n;

            while (rand_ind >= RAND_MAX - (RAND_MAX % n)) {
                rand_ind = rand();
            }
            rand_ind %= n;

            features.push_back( rand_ind  );
        }

        return features;
    }

    Node generate_root_node(data_t &data, vector<range_t> &ranges) {
        Node root = Node();
        root.data = data;
        root.ranges = ranges;
        root.memberships = vector<double>(data.size(), 1.0);
        root.f = function<double(item_t)>([](item_t i) { return 1.0; });

        fill_node_properties(&root, &root);

        return root;
    }
};


class RandomFuzzyForest {
private:
    vector<RandomFuzzyTree> classifiers;
    int job_n;
    data_t data;
    vector<range_t> ranges;
public:
    RandomFuzzyForest(int classifier_n, int job_n=4) {
        classifiers = vector<RandomFuzzyTree>(classifier_n);
        this->job_n = job_n;
    }

    void fit(data_t &data, vector<range_t> &ranges) {
        double perc = 1.0;

        data_t classifier_data;
        data_t weight_data;

        for(int i = 0; i < data.size(); i++) {
            auto &d = data[i];
            if(i < perc * data.size()) {
                classifier_data.push_back(d);
            } else {
                weight_data.push_back(d);
            }
        }

        fit_classifiers(classifier_data, ranges);
        fit_weights(weight_data);
    }

    void fit_classifiers(data_t &data, const vector<range_t> &ranges) {
        vector<thread> threads(classifiers.size());

        for(int i = 0; i < ceil((double) classifiers.size() / job_n); i++) {
            for(int j = 0; j < job_n; j++) {
                int curr_ind = i * job_n + j;
                if(curr_ind < classifiers.size()) {
                    threads[j] = thread( [this, data, ranges, curr_ind] {
                        fit_classifier(&classifiers[curr_ind],
                                       data,
                                       ranges);
                    } );
                }
            }

            for(int j = 0; j < job_n; j++) {
                int curr_ind = i * job_n + j;
                if(curr_ind < classifiers.size()) {
                    threads[j].join();
                }
            }
        }
    }

    void fit_weights(data_t &data) {
        double alpha = 0;

        for(auto &d : data) {
            string y = d.second;

            for(auto &classifier : classifiers) {
                map<string, double> prediction = classifier.predict_memberships(d);

                double total = 0;
                auto &weights = classifier.weights();
                for(auto &kv : prediction) {
                    total += weights[kv.first] * kv.second;
                }

                for(auto kv : weights) {
                    double t;
                    if(kv.first.compare(y) == 0) {
                        t = total;
                    } else {
                        t = 0;
                    }

                    double dy = weights[kv.first] * prediction[kv.first] - t;
                    double delta_w = -alpha * dy * prediction[kv.first];

                    weights[kv.first] += delta_w;
                }
            }
        }
    }

    void fit_classifier(RandomFuzzyTree *classifier,
                        data_t data,
                        vector<range_t> ranges) {
        data_t data_sample = random_sample(data);
        classifier->fit(data_sample, ranges);
    }

    data_t random_sample(data_t &data) {
        set<item_t> sample;
        for(int i = 0; i < data.size(); i++) {
            int rand_ind = rand();
            int n = (int) data.size();

            while (rand_ind >= RAND_MAX - (RAND_MAX % n)) {
                rand_ind = rand();
            }
            rand_ind %= n;

            sample.insert(data[rand_ind]);
        }

        data_t sample_data;
        copy(sample.begin(), sample.end(), back_inserter(sample_data));

        return sample_data;
    }

    string predict(item_t &x) {
        map<string, double> membs_per_class;

        for(int i = 0; i < classifiers.size(); i++) {
            auto classifier = classifiers[i];
            map<string, double> membs = classifier.predict_memberships(x);
            for(auto kv : membs) {
                string cls = kv.first;
                double val = kv.second;

                membs_per_class[cls] += val * classifier.weights()[cls];
            }
        }

        return std::max_element(membs_per_class.begin(),
                                membs_per_class.end(),
                                [](const pair<string, int>& p1, const pair<string, int>& p2) {
                                    return p1.second < p2.second; })->first;
    }

    double score(data_t &data) {
        int correct_n = 0;
        for(item_t &d : data) {
            string prediction = predict(d);
            if(prediction == d.second) {
                correct_n++;
            }
        }

        return (double)correct_n / data.size();
    }

};

int main() {
    auto string_data = load_csv_data("/home/faruk/workspace/thesis/data/segmentation.dat",
                                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                                     0,
                                     2100);
    auto ranges = find_ranges(string_data,
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});

    data_t data;
    for(auto &x : string_data) {
        vector<double> item;
        for(auto val : x.first) {
            item.push_back(to_double(val));
        }
        string classification = x.second;
        data.push_back(make_pair(item, classification));
    }

    bool shuffle = true;
    if(shuffle) {
        random_shuffle(data.begin(), data.end());
    }

    int clasifier_n = 100;
    int job_n = 4;

    int fold_n = 10;
    int per_fold = (int) (data.size() / fold_n);

    RandomFuzzyForest rff(clasifier_n, job_n);

    double total_score = 0;
    for(int i = 0; i < fold_n; i++) {
        data_t training_data;
        data_t verification_data;

        cout << "Scoring fold " << i << endl;
        for(int j = 0; j < data.size(); j++) {
            if(j >= i * per_fold and j < (i + 1) * per_fold) {
                verification_data.push_back(data[j]);
            } else {
                training_data.push_back(data[j]);
            }
        }

        rff.fit(training_data, ranges);
        double curr_score = rff.score(verification_data);
        cout << "\tScore: " << curr_score << endl;

        total_score += curr_score;
    }

    cout << "Total average score: " << total_score / fold_n << endl;

    return 0;
}