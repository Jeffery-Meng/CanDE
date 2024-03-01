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
    cout << "Usage: ./falconn_recall_table [config file]";
  }

  LSHConstructionParameters conf = read_config(argv[1]);
  int num_tables = conf.hash_table_params[0].l;
  int max_num_tables = conf.hash_table_params[1].l;
  std::vector<int> recalls(max_num_tables, 0), candidates(max_num_tables, 0);

  auto gt_orders = read_data<DenseVector<int>>(conf.gnd_filename);
  std::vector<HashSet<int>> gt_sets(conf.num_queries);
  for (int qid = 0; qid < conf.num_queries; ++qid) {
    // when reverse orders are used
    // auto begin_iter = gt_orders[qid].end() - conf.num_neighbors;
    // auto end_iter = gt_orders[qid].end();
    auto begin_iter = gt_orders[qid].begin();
    auto end_iter = gt_orders[qid].begin() + conf.num_neighbors;
    gt_sets[qid].insert(begin_iter, end_iter);
  }

  for (int qid = 0; qid < conf.num_queries; ++qid) {
    std::string candidate_file = conf.candidate_filename + "_q_" + std::to_string(qid) + ".bin";
    std::ifstream fin(candidate_file, std::ios::binary);

    unsigned char mode;
    int num_candidates;

    fin.read(reinterpret_cast<char*> (&mode), sizeof(mode));
    bool read_bucket = (mode == 0b11100000);
    assert(mode == 0b11000000 || read_bucket);
    fin.read(reinterpret_cast<char*> (&num_candidates), sizeof(int));
    for (int cid = 0; cid < num_candidates; ++cid) {
      int id, table, bucket;
      fin.read(reinterpret_cast<char*> (&id), sizeof(int));
      fin.read(reinterpret_cast<char*> (&table), sizeof(int));
      if (read_bucket) {
        fin.read(reinterpret_cast<char*> (&bucket), sizeof(int));
      }
      if (table >= max_num_tables) continue;
      if (gt_sets[qid].contains(id)) {
        for (int tid = table; tid < max_num_tables; ++tid) {
          ++recalls[tid];
        }
        gt_sets[qid].erase(id);
      }
      for (int tid = table; tid < max_num_tables; ++tid) {
          ++candidates[tid];
      }
    }
  }

  double denom = 1.0 / conf.num_queries / conf.num_neighbors;
  double denom_candidates = 1.0 / conf.num_queries / conf.num_points;
  std::string output_name = conf.summary_path + "recalls.txt";
  std::ofstream fout(output_name);
  for (auto val : recalls) {
    fout << val * denom << "\t";
  }
  fout << endl;
  for (auto val : candidates) {
    fout << val * denom_candidates << "\t";
  }
  fout << endl;
}