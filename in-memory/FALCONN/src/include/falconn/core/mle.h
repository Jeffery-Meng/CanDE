#ifndef __MLE_H__
#define __MLE_H__

unsigned max_estimate = 1000000;
#include <vector>
#include <algorithm>

namespace falconn {
namespace core {
double mle_func(unsigned x,std::vector<unsigned> num_vec,std::vector<double> prob_vec)
{
    double prob_temp = 1.0;
    double target_temp = 1.0;
    for (int i=0;i<num_vec.size();i++)
    {
        prob_temp *= (1.0-1.0*num_vec[i]/x);
        target_temp *= (1.0-prob_vec[i]);
    }
    return prob_temp - target_temp;
}

// Calculate the estimated number of points based on MLE, input is probability vector and number in each hash table vector
// num_vecs: number of points in each hash table; prob_vecs: probability vector 
unsigned gen_esimation(std::vector<unsigned> num_vec,std::vector<double> prob_vec)
{
   std::vector<unsigned> check_vec(num_vec.size(), 0);
   if (num_vec == check_vec){
       return 0;}
    else {
        unsigned min = *std::max_element(num_vec.begin(),num_vec.end());
        unsigned max = max_estimate;
        auto f_min= mle_func(min,num_vec,prob_vec);
        while(max - min <= 1) {
            unsigned mid = static_cast<unsigned int>(0.5*min + 0.5*max);
            auto f_mid = mle_func(mid,num_vec,prob_vec);
            if ((f_min < 0) == (f_mid < 0)) {
            min = mid;
            f_min = f_mid;
        } else {
            max = mid;
        }   
        }
        return min;
    }
}

}}
#endif