#include <iostream>
#include <string>

using namespace std;


int main(int argc, char **argv) {

    int REDUCER_N = stoi(getenv("REDUCERN"));

    string line;
    int i = 0;
	while (getline(cin, line)) {
        if(line.find("?") == std::string::npos) {
            if(line[0] == 't') {
                for(int j = 0; j < REDUCER_N; j++) {
                    cout << j << "\t" << line << endl;
                }
            } else {
                cout << i << "\t" << line << endl;
            }
        }

        i = (i + 1) % REDUCER_N;
	}

    return 0;
}