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


int main(int argc, char **argv) {

    int REDUCER_N = stoi(getenv("REDUCERN"));
    int K = 5;
    srand(time(NULL));

    string line;
    int i = 0;
	while (getline(cin, line)) {
        if(line.find("?") == std::string::npos) {
            if(line[0] == 't') {
                for(int j = 0; j < REDUCER_N; j++) {
                    cout << j << "\t" << line << endl;
                }
            } else {
                for(int j = 0; j < K; j++) {
                    cout << ( rand() % REDUCER_N ) << "\t" << line << endl;
                }
            }
        }

        i = (i + 1) % REDUCER_N;
	}

    return 0;
}