#include <iostream>
#include "rounding_header.hpp"
#include <fstream>

using namespace std;

int main(int argc, char * argv[]) {
    RoundingTestbed<JXRounding> rounding;
    int num_exp = 10000, sketch_sz = 10000;
    float near = 32.0, far = 32.32;
    ofstream fout(argv[1]);

    for (int ii = 0; ii < num_exp; ++ii) {
        fout << rounding.sketch(near, sketch_sz) << "\t" ;
    }
    fout << endl;
    for (int ii = 0; ii < num_exp; ++ii) {
        fout << rounding.sketch(far, sketch_sz) << "\t" ;
    }
    fout << endl;

    return 0;
}