#ifndef __ONIAK_RECALL_HPP__
#define __ONIAK_RECALL_HPP__

#include <algorithm>
#include <cassert>
#include <vector>

namespace ONIAK {

// assumes the results and gts are sorted for each query
using IntegerMatrix = std::vector<std::vector<int>>;
std::vector<double> recall_all_q(const IntegerMatrix& results, const IntegerMatrix& ground_truth, size_t k) {
  int qn = results.size();
  std::vector<double> recall(qn);
  for (int qid = 0; qid < qn; ++qid) {
    const auto& result = results[qid];
    const auto& gt = ground_truth[qid];
    assert(gt.size() == k);
    std::vector<int> v_intersection;
    std::set_intersection(gt.begin(), gt.end(), result.begin(), result.end(),
                          std::back_inserter(v_intersection));
    recall[qid] = v_intersection.size() / static_cast<double>(k);
  }
  return recall;
}

std::vector<std::vector<double>> QDDE_gt(const falconn::MatrixType& distance, const falconn::FalconnRange& bins,
                                    std::vector<int>& selected_queries) {
  int num_queries = distance.rows();
  int num_data = distance.cols();
  int num_bins = bins.num_bins();
  std::vector<std::vector<double>> ground_truth(num_bins,
                                                std::vector<double>(num_queries, 0.0));
  for (int qid = 0; qid < num_queries; ++qid) {
    for (int id = 0; id < num_data; ++id) {
      ground_truth[bins.bin_translate(distance(qid, id))][qid] += 1.0;
    }
  }

  // selected queries are those without empty bins.
  selected_queries.clear();
  for (int qid = 0; qid < num_queries; ++qid) {
    bool no_empty_bin = true;
    // The first and last bins are out of range.
    for (int bid = 1; bid < num_bins-1; ++bid) {
      if (ground_truth[bid][qid] == 0) {
        no_empty_bin = false;
        break;
      }
    }
    if (no_empty_bin) {
      selected_queries.push_back(qid);
    }
  }
  return ground_truth;
}

} // namespace ONIAK

#endif