#include "falconn/config.h"
#include "falconn/fileio.h"
#include "falconn/kronecker/utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "boost/unordered_set.hpp"

/* Compute the retrival probablity of each candidate. */

using namespace falconn;
using namespace std;
template <typename T>
using HashSet = boost::unordered_set<T>;

int id_translate(float val) {
  int result = val * 10000;
  if (result < 0) result = 0;
  else if (result >= 30000) result = 29999;
  return result;
}

double final_prob(float recall, int num_tables) {
  return 1 - std::pow(1.0 - recall, num_tables);
}

int main(int argc, char * argv[]) {
  if (argc < 2) {
    cout << "Usage: ./candidate_probs [config file]";
  }


  int fast_stop = std::numeric_limits<int>::max();
  if (argc >= 3) {
    fast_stop = atoi(argv[2]);
  }

   
  LSHConstructionParameters conf = read_config(argv[1]);
  int num_tables = conf.hash_table_params[0].l;
  float bucket_width = conf.hash_table_params[0].bucket_width;

  auto distances_raw = read_data<VectorType>(conf.distance_filename);
  auto query_hash_raw = read_data<VectorType>(conf.query_hash_filename);
  MatrixType distances(conf.num_queries, conf.num_points);
  for (int row = 0; row < conf.num_queries; ++row) {
    distances.row(row) = distances_raw[row];
  } 
  VectorType probing_probs = read_data<VectorType>(conf.mp_prob_filename)[0];

  std::vector<std::vector<int>> candidates(conf.num_queries);
  for (int qid = 0; qid < conf.num_queries; ++qid) {
    std::string candidate_file = conf.result_filename + "_q_" + std::to_string(qid) + ".bin";
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
      candidates[qid].push_back(id);
    }
  }

  std::vector<std::vector<double>> recall_ps(conf.num_queries);
  for (int qid = 0; qid < conf.num_queries; ++qid) {
    if (qid > fast_stop) break;

    for (int id : candidates[qid]) {
      std::vector<double> recalls(num_tables, 0);
      double relative_distance = distances(qid, id) / bucket_width;
      int distance_id = id_translate(relative_distance);
      double recall_per_table = probing_probs[distance_id];
      double final_recall = final_prob(recall_per_table, num_tables);
      recall_ps[qid].push_back(final_recall);
    }
  }

  std::ofstream fout(conf.recall_p_filename);
  for (auto& recall : recall_ps) {
    write_data(fout, recall);
  }

  return 0;
}