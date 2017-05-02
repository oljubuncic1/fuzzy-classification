#include <iostream>
#include <string>

using namespace std;

#define REDUCER_N 3

int main() {

    string line;
	while (getline(cin, line)) {
        if(line.find("?") == std::string::npos) {
            if(line[0] == 't') {
                for(int j = 0; j < REDUCER_N; j++) {
                    cout << j << "\t" << line << endl;
                }
            } else {
                cout << (rand() % REDUCER_N) << "\t" << line << endl;
            }
        }
	}

    return 0;
}