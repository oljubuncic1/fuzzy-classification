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

struct ReducerGroup {
    vector<int> ids;
    int current;

    ReducerGroup() {
        current = 0;
    }

    void assign_item(string &line) {
        if(line[0] == 't') {
            cout << current << "\t" << line << endl;
            current = (current + 1) % ids.size();
        } else {
            for(auto id : ids) {
                cout << id << "\t" << line << endl;
            }
        }
    }
};

int main(int argc, char **argv) {

    int REDUCERS_PER_GROUP = stoi(getenv("PERGROUP"));
    int GROUP_N = stoi(getenv("GROUPN"));
    int K = stoi(getenv("K"));
    srand(time(NULL));

    vector<ReducerGroup> groups(GROUP_N);
    
    int curr = 0;
    for(int i = 0; i < GROUP_N; i++) {
        for(int j = 0; j < REDUCERS_PER_GROUP; j++) {
            groups[i].ids.push_back(curr);
            curr++;
        }
    }

    string line;
	while (getline(cin, line)) {
        if(line.find("?") == std::string::npos) {
            groups[ rand() % groups.size() ].assign_item(line);
        }
    }

    return 0;
}