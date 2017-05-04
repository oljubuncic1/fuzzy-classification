#include <iostream>
#include <string>

using namespace std;


int main(int argc, char ** argv) {

    int REDUCER_N = stoi(argv[1]);

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