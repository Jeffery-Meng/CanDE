#include "falconn/config.h"
#include "falconn/fileio.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "boost/unordered_set.hpp"

using namespace falconn;
using namespace std;

int main(int argc, char * argv[]) {
  if (argc < 2) {
    cout << "Usage: ./falconn_selectivity [config file]";
  }

  LSHConstructionParameters conf = read_config(argv[1]);
  auto candidates = read_candidates(conf);
  int64_t cnt = 0;
  for (auto& cv : candidates) {
    cnt += cv.size();
  }
  double selectivity = static_cast<double>(cnt) / conf.num_queries / conf.num_points;
  std::string output_name = conf.summary_path + "selectivity.txt";
  std::ofstream fout(output_name);
  fout << conf.dataset << "\t" << selectivity << std::endl;
  return 0;
}