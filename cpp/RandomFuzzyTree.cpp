#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <armadillo>
#include <set>

using namespace std;
using namespace arma;

struct FuzzyNode;

struct FuzzySetProperties {
    double cardinality;
    double entropy;
    mat data;
    vec memberships;
};

struct FuzzyPartition {
    function<double(double)> f;
    FuzzyNode* node;
    vector<FuzzySetProperties*> properties;
};

struct FuzzyPartitioning {
    vector<FuzzyPartition*> partitions;
    double gain;
};

struct FuzzyNode {
    bool is_terminal;
    string classification;
    int feature;
    FuzzyPartitioning* partitioning;
};


class RandomFuzzyTree {
private:
    int n_jobs;
    int p;
    bool is_fit;
    double terminal_n_threshold;
    set<int> classes;
public:
    RandomFuzzyTree(int n_jobs=1,
                            int p=5,
                            double terminal_n_threshold=10) {

        this->n_jobs = n_jobs;
        this->p = p;
        this->is_fit = false;
        this->terminal_n_threshold = terminal_n_threshold;
    
    }

        

//     def fit(self, data, ranges, copy_data=False, classes=(1, 2)):
//         self.classes = classes

//         if copy_data:
//             data = np.copy(data)

//         self.ranges = ranges
//         self.n_feature = self.count_features(data)
//         tree = self.build_tree(data, np.array([1.0 for d in data]))
//         self.root = tree

//         self.is_fit = True

//     def predict(self, x):
//         raise NotImplementedError()

//     def score(self, data):
//         raise NotImplementedError()

//     def build_tree(self, data, memberships, lvl=0):
//         print(lvl)
//         node = FuzzyNode()

//         if self.is_terminal(data, memberships):
//             node.is_terminal = True
//             node.classification = self.classification(data, memberships)

//             return node
//         else:
//             regular_features = []
//             for i in range(len(self.ranges)):
//                 curr_range = self.ranges[i]
//                 if curr_range[0] != curr_range[1]:
//                     regular_features.append(i)

//             features = np.random.choice(regular_features,
//                                         self.p,
//                                         replace=False)

//             feature_partitionings = {}
//             for feature in features:
//                 feature_partitionings[feature] = \
//                     self.best_partitioning(feature, data, memberships)

//             node.feature = max(feature_partitionings,
//                                key=lambda x: feature_partitionings[x].gain)
//             node.partitioning = feature_partitionings[feature]

//             for p in node.partitioning.partitions:
//                 p.node = self.build_tree(p.properties.data, p.properties.memberships, lvl + 1)

//             return node

//     def is_terminal(self, data, memberships):
//         return np.sum(memberships) < 10

//     @staticmethod
//     def count_features(data):
//         return data.shape[1] - 1

//     def classification(self, data, memberships):
//         return 1
//         raise NotImplementedError()

//     def best_partitioning(self, feature, data, memberships):
//         points = np.unique(data[:, feature])
//         L, U = self.ranges[feature]

//         point_partitionings = {}
//         regular_point_occured = False
//         i = 0
//         for p in points:
//             print("\t", i)
//             i += 1
//             if p != L and p != U:
//                 regular_point_occured = True
//                 point_partitionings[p] = \
//                     self.partitioning(data, feature, p, memberships)

//         if not regular_point_occured:
//             midpoint = L + (U - L) / 2
//             max_partitioning = self.partitioning(data,
//                                                  feature,
//                                                  midpoint,
//                                                  memberships)
//         else:
//             max_partitioning_key = max(point_partitionings,
//                                        key=lambda x: point_partitionings[x].gain)

//             max_partitioning = point_partitionings[max_partitioning_key]

//         return max_partitioning

//     def partitioning(self, data, feature, p, memberships):
//         part = FuzzyPartitioning()

//         L, U = self.ranges[feature]
//         W_left = 2 * (p - L)
//         W_middle_left = (p - L)
//         W_middle_right = (U - p)
//         W_right = 2 * (U - p)

//         # TODO: generalize to more
//         left_partition = FuzzyPartition()
//         left_partition.f = math_functions.triangular(L,
//                                                      W_left)
//         left_partition.properties = []

//         middle_partition = FuzzyPartition()
//         middle_partition.f = \
//             math_functions.composite_triangular(p,
//                                                 W_middle_left,
//                                                 W_middle_right)

//         right_partition = FuzzyPartition()
//         right_partition.f = math_functions.triangular(U,
//                                                       W_right)

//         part.partitions = [left_partition,
//                            middle_partition,
//                            right_partition]

//         self.set_properties(part.partitions, data, feature, memberships)
//         part.gain = self.gain(part.partitions, memberships)

//         return part

//     def set_properties(self, partitions, data, feature, memberships):
//         for partition in partitions:
//             prop = self._fuzzy_set_properties(data, feature, partition, memberships)
//             partition.properties = prop

//     def gain(self, partitions, memberships):
//         data_cardinality = np.sum(memberships)
//         if len(partitions) == 0:
//             raise ValueError("Empty partitions")
//         properties = [part.properties for part in partitions]
//         gain_value = 0
//         for prop in properties:
//             gain_value -= prop.cardinality / data_cardinality * prop.entropy

//         return gain_value

    FuzzySetProperties _fuzzy_set_properties(mat &data, 
                                             int feature, 
                                             FuzzyPartition &partition, 
                                             vec &memberships) {
        if(data.n_elem == 0) {
            throw invalid_argument("Empty array");
        }

        vec set_memberships = data.col(feature);

        vec::iterator a = set_memberships.begin();
        vec::iterator b = set_memberships.end();

        for(vec::iterator i = a; i != b; i++) {
            (*i) = partition.f((double)*i);
        }

        set_memberships = memberships % set_memberships;
        double cardinality = accu(set_memberships);

        double entropy = this->_fuzzy_entropy(data,
                                              set_memberships,
                                              cardinality);

        auto properties = FuzzySetProperties();
        properties.cardinality = cardinality;
        properties.entropy = entropy;

        auto non_zero_inds = find(set_memberships);
        auto set_data = data.elem(non_zero_inds);
        vec next_set_memberships = set_memberships.elem(non_zero_inds);

        properties.data = &set_data;
        properties.memberships = &next_set_memberships;

        return properties;
    }

    double _fuzzy_entropy(mat &data, vec &memberships, double cardinality=-1) {
        if(data.n_elem == 0) {
            throw invalid_argument("Empty array");
        }

        double entropy = 0;
        if(cardinality == -1) {
            cardinality = accu(memberships);
        }
        
        if(cardinality != 0) {
            auto data_classes = data.col(-1);
            for(auto c : this->classes) {
                auto inds = find(data_classes == c);
                auto memberships_at_inds = memberships.elem(inds);
                double proba = accu(memberships_at_inds) / cardinality;
                if(proba != 0) {
                    entropy -= proba * log2(proba);
                }
            }
                
        }

        return entropy;
    }

//     def __str__(self):
//         raise NotImplementedError()
};

int main() {
    return 0;
}