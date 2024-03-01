#include "falconn/config.h"
#include "falconn/falconn_global.h"
#include "falconn/fileio.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>

#include "boost/unordered_set.hpp"
#include "boost/unordered_map.hpp"

#include <Timer.hpp>

using namespace falconn;
using namespace std;
using namespace ONIAK;



int main(int argc, char * argv[]) {
  if (argc < 2) {
    cout << "Usage: ./qdde_gt [config file]";
    exit(1);
  }

  LSHConstructionParameters conf = read_config(argv[1]);
  std::string historgram_filename = conf.histogram_filename;
  cout << "writing histogram to " << historgram_filename << endl;
  auto distances_raw = read_data<VectorType>(conf.distance_filename);
  MatrixType distances(conf.num_queries, conf.num_points);
  for (int row = 0; row < conf.num_queries; ++row) {
    distances.row(row) = distances_raw[row];
  }
  distances_raw.clear();


  falconn::falconn_config = std::move(conf);
  FalconnRange& range = falconn_config.bins_vector[0];
  int num_bins = range.num_bins();


  vector<vector<int>> results;
  for (int qid = 0; qid < conf.num_queries; ++qid) {
      vector<int> row_result(num_bins, 0);
      for (auto cur_distance : distances.row(qid)) {
        int bin = range.bin_translate(cur_distance);
        row_result[bin] += 1;
        }
        results.push_back(row_result); 
      }
    
    cout << "result size: " << results.size() << endl;
    std::ofstream fout(historgram_filename);
    for (auto& result_temp : results) {
      write_data(fout, result_temp);
    }

    return 0;
  }
