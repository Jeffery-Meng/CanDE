#ifndef __FALCONN_HISTOGRAM_H__
#define __FALCONN_HISTOGRAM_H__


 auto bin_translate = [&](double distance) -> int {
      distance = (distance > bin_end)? bin_end : distance;
      int idx = std::floor((distance - bin_start) / bin_step) + 1;
      idx = (idx < 0)? 0: idx;
      return idx;
    };
    int num_bins = bin_translate(bin_end) + 1;

#endif