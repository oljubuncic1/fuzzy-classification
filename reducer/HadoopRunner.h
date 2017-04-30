//
// Created by faruk on 4/30/17.
//

#ifndef CPP_HADOOPRUNNER_H
#define CPP_HADOOPRUNNER_H

#include "includes.h"
#include "definitions.h"
#include "data.h"
#include "definitions.h"
#include "RandomFuzzyTree.h"
#include "RandomFuzzyForest.h"

#define NONE_WORD "EMPTY"

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


class HadoopRunner {
public:
    void run() {

    }
};

#endif //CPP_HADOOPRUNNER_H
