#ifndef __SUCC_PROBABILITY_H__
#define __SUCC_PROBABILITY_H__

#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include "heap.h"


namespace falconn {
namespace core {

template <typename distance_type>
class gauss_prob {
    gauss_prob(float bucket_width,int num_hash_func,int probes_each_table)
    :bucket_width_(bucket_width),
    k_(num_hash_func),
    probes_each_table_(probes_each_table),
    stay_epic_prob_(0){
        // Read Gaussian CDF
        std::ifstream infile("gaussian_cdf.txt");
        double temp1;
        double temp2;
        while(infile >> temp1 >> temp2){
            gauss_cdfs.insert({temp1,temp2});
    }

    // Generate the precomputed perturbation vector
    SimpleHeap<double,std::vector<int_fast32_t>> heap_temp;  
      // insert the best perturbation vector (stored in hash_mask) of every hash table
      // score is the precomputed value
      std::vector<int_fast32_t> temp_pert;
      temp_pert.push_back(1);
      double cur_score = 1*(1+1)*1.0/((4*(k_+1)*(k_+2))*1.0*(bucket_width_*bucket_width_));
      double temp_score;
      heap_temp.reset();
      heap_temp.insert_unsorted(cur_score,temp_pert);
      heap_temp.heapify();

      for(int_fast32_t i =0; i< probes_each_table;i++){
        int label = 0;
        while(label != 1) {
           heap_temp.extract_min(&cur_score, &temp_pert);
           // Shift on the vector
           std::vector<int_fast32_t> temp_shift_pert;
           temp_score = 0;
           for(int j=0;j<temp_pert.size();j++) {
             temp_shift_pert.push_back((j<temp_pert.size()-1)? temp_pert[j] : temp_pert[j]+1);
             temp_score += (temp_shift_pert.back()<=k_)? 1.0*temp_shift_pert.back()*(temp_shift_pert.back()+1)/(4*(k_+1)*(k_+2)*1.0)*1.0*(bucket_width_*bucket_width_): 
              1.0*(bucket_width_*bucket_width_)*(1-(2*k_+1-temp_shift_pert.back())*1.0/(k_+1)+(2*k_+1-temp_shift_pert.back())*(2*k_+2-temp_shift_pert.back())*1.0/(4*(k_+1)*(k_+2)));
           }
          heap_temp.insert_unsorted(temp_score,temp_shift_pert);
          heap_temp.heapify();

          // Expand the vector
          std::vector<int_fast32_t> temp_expand_pert;
          temp_score = 0;
          for(int j=0;j<=temp_pert.size();j++) {

             temp_expand_pert.push_back((j<temp_pert.size())? temp_pert[j] : temp_pert[j-1]+1);
             temp_score += (temp_expand_pert.back()<=k_)? 1.0*temp_expand_pert.back()*(temp_expand_pert.back()+1)/(4*(k_+1)*(k_+2)*1.0)*1.0*(bucket_width_*bucket_width_): 
              1.0*(bucket_width_*bucket_width_)*(1-(2*k_+1-temp_expand_pert.back())*1.0/(k_+1)+(2*k_+1-temp_expand_pert.back())*(2*k_+2-temp_expand_pert.back())*1.0/(4*(k_+1)*(k_+2)));
           }
          heap_temp.insert_unsorted(temp_score,temp_expand_pert);
          heap_temp.heapify();
          // Verify the vector
          int temp_label = 1;
          for(int i=0;i<=temp_pert.size();i++) {
              if (contains(temp_pert,temp_pert[i]) && contains(temp_pert,2*k_+1-temp_pert[i]))
              {
               temp_label = 0;
              }
            }
        if (temp_pert.back() > 2*k_) temp_label =0;
        if (temp_label == 1) break;
        }
        probes_vecs_.push_back(temp_pert);
      }
    }

    double cdf_new(float hash_distance, distance_type d) {
        double temp = hash_distance/sqrt(d);
        if (temp>= 8.0) 
        {
            return 1.0;
        } else if (temp <= -8) {
            return 0.0;
        } else {
            return gauss_cdfs[temp];
        }

    }

    // // Probabilities that stay in the epicenter bucket
    // void gen_stay_epic_pros(distance_type d,std::vector<float> hash_distance) {
    //     double stay_prob = 1.0;
    //     for (auto distance_val:hash_distance)
    //     {
    //         double prob_temp = gen_stay_prob(d,distance_val);
    //         stay_prob *= prob_temp;
    //         stay_probs.push_back(prob_temp);
    //     } 
    //     stay_epic_prob_ = stay_prob;
    // }

    //  Probabilty that stay in the i-th bucket
    double gen_stay_prob(distance_type d,float hash_distance_val) {
        return cdf_new(hash_distance_val,d) - cdf_new(-(bucket_width_-hash_distance_val),d);
    }

    // Probability that distribution vector will exceed the bucket
    double gen_pro_distribution(distance_type d,float hash_distance_val) {
        return cdf_new(hash_distance_val+bucket_width_, d) - cdf_new(hash_distance_val, d);
    }

    //Probabilities that p in certain perturbation vector 
    double pro_dist(std::vector<int_fast32_t> pert_vec, std::vector<float> hash_distance, distance_type d){
        std::vector<int> total_ele(k_);
        std::iota(total_ele.begin(),total_ele.end(),0);
        double pro_temp = 1.0;
        for (auto item: pert_vec){
            if (item<=2*k_) 
                pro_temp *= gen_pro_distribution(d,hash_distance[item-1]);
            else { return 0.0;}

            // Remove the item from total elements
            if (item > k_){
                if ( std::find(total_ele.begin(), total_ele.end(), 2*k_+1-item) != total_ele.end()) 
                {total_ele.erase(std::remove(total_ele.begin(), total_ele.end(), 2*k_+1-item), total_ele.end());}
                else {return 0.0;}
            }else {total_ele.remove(item);}
        }
                
        // Now multiply the probability of stay in each bucket
        for (auto item: total_ele){
            pro_temp = pro_temp * stay_probs[item-1];
        }
        return pro_temp;
    }

    // Generate the cdf of precomputed perturbation set
    def gen_cdf(std::vector<float> hash_distance, distance_type d) {
        double stay_prob = 1.0;
        for (auto distance_val:hash_distance)
        {
            double prob_temp = gen_stay_prob(d,distance_val);
            stay_prob *= prob_temp;
            stay_probs.push_back(prob_temp);
        } 
        stay_epic_prob_ = stay_prob;

        double prob = stay_epic_prob_; // Original probablities, what LSH cares
        for (auto pert_vec: probes_vecs_)
        {
            prob = prob + pro_dist(pert_vec,hash_distance,d);
        }
        return prob;
    }
    public:
        float bucket_width_;
        int_fast32_t k_, probes_each_table_;
        double stay_epic_prob_;
        std::vector<double> stay_probs;
        std::vector<std::vector<int_fast32_t>> probes_vecs_;
        std::unordered_map<double,double> gauss_cdfs;
};

}}

#endif