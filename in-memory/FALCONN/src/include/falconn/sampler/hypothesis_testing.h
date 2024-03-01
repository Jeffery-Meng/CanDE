#ifndef __HYPOTHESIS_TESTING_H__
#define __HYPOTHESIS_TESTING_H__

#include <optional>
#include "../falconn_global.h"
#include <boost/math/distributions/chi_squared.hpp>

// Utility functions for hypothesis testing.

namespace falconn {

HashMap<int, int> histogram_from_samples(const std::vector<int>& fair_samples,
  int& num_failures) {
  num_failures = 0;
  HashMap<int, int> frequency_cnt;
    for (int sample : fair_samples) {
      if (sample == -1){
        ++num_failures;
        continue;  // failure
      }
      if (frequency_cnt.find(sample) == frequency_cnt.end()) {
        frequency_cnt[sample] = 1;
      } else {
        ++frequency_cnt[sample];
      }
    }
    return frequency_cnt;
}

HashMap<int, int> histogram_from_samples(const std::vector<int>& fair_samples) {
  int dummy;
  return histogram_from_samples(fair_samples, dummy);
}

double total_variation(const HashMap<int, int>& frequency_cnt, int num_samples, int num_neighbors) {
    double tv_value = 0.0;
    double average_cnt = static_cast<double>(num_samples) / num_neighbors;
    for (auto val : frequency_cnt) {
      double cnt = val.second;
      tv_value += std::abs(average_cnt - cnt);
    }
    // points that are not returned
    tv_value += average_cnt * (num_neighbors - frequency_cnt.size());
    return tv_value / num_samples;
}

double chi_square_test(const HashMap<int, int>& frequency_cnt, int num_samples, int num_neighbors) {
    double cs_value = 0.0;
    double average_cnt = static_cast<double>(num_samples) / num_neighbors;
    for (auto val : frequency_cnt) {
      double cnt = val.second;
      cs_value += (average_cnt - cnt) * (average_cnt - cnt) / average_cnt;
    }
    // points that are not returned
    cs_value += average_cnt * (num_neighbors - frequency_cnt.size());
    return cs_value;
}

// Converts chi-square statistics to p-value
double chi_square_p_value(double chi2, int num_neighbors) {
  // Degree of freedom is num_neighbors - 1
  boost::math::chi_squared chi2_dist(num_neighbors - 1);
  return boost::math::cdf(chi2_dist, chi2);
}

}

#endif