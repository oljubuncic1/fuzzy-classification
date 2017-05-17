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
    srand(time(NULL));

    string line;
    int i = 0;
    int FOLD_N = 5;
    int EVERY_N_LINES = 5;

    while (getline(cin, line)) {
        if(line.find("?") == string::npos && rand() % EVERY_N_LINES == 0) {
            if(rand() % FOLD_N == 0) {
                cout << "t" << line << endl;
            } else {
                cout << line << endl;
            }
        }
    }

    return 0;
}