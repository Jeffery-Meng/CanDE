#include "../cpu_timer.hpp" 
#include <iostream>

using namespace std;

int main() {
    CPUTimer timer;
    int i = 0;
    float a = 0.;
    cout << timer.watch() << endl;

    timer.start();
    for(; i < 1e9; ++i) {
        a = a + i * i;
    }
    timer.stop();
    cout << a << endl;
    cout << timer.watch() << endl;
}
