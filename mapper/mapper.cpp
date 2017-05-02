#include <iostream>
#include <string>

using namespace std;

#define REDUCER_N 3

int main() {

    string line;
    int i = 0;
	while (getline(cin, line)) {
        if(find(line.begin(), line.end(), '?') == line.end()) {
            if(line[0] == 't') {
                for(int j = 0; j < REDUCER_N; j++) {
                    cout << j << "\t" << line << endl;
                }
            } else {
                cout << i << "\t" << line << endl;
            }
            i = (i + 1) % REDUCER_N;
        }
	}

    return 0;
}