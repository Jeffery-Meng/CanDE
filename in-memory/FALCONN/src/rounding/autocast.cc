#include <iostream>
#include "rounding_header.hpp"

using namespace std;


template <auto Start, auto End, auto Inc>
constexpr void initialize_rf()
{
    if constexpr (Start < End)
    {
        RoundedFloat<Start>::scale = 16.;
        initialize_rf<Start + Inc, End, Inc>();
    }
}


int main() {
    constexpr int partition_max = 10;
    initialize_rf<0, partition_max, 1>;
    RoundedFloat<0> a = 9;
   // RoundedFloat<0>::scale = 16;
    cout << (float) a << endl;
    return 0;
}