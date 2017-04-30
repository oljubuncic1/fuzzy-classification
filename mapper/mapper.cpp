#include <iostream>
#include <string>

using namespace std;

#define REDUCER_N 5;

int main() {

    string line;
    int i = 0;
	while (getline(cin, line)) {
		cout << i << "\t" << line << endl;
        i = (i + 1) % REDUCER_N;
	}

    return 0;
}