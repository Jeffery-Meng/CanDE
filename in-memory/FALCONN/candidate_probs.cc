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
  int result = (val + 5) * 1000000;
  if (result < 0) result = 0;
  else if (result >= 10000000) result = 9999999;
  return result;
}

double bucket_prob(const VectorType& gaussian_cdf, const VectorType& lower,
    const VectorType& upper) {
    double result = 1.0;
    int sz = lower.size();
    for (int i = 0; i < sz; ++i) {
      int lower_id = id_translate(lower(i));
      int upper_id = id_translate(upper(i));
      result *= gaussian_cdf[upper_id] - gaussian_cdf[lower_id];
    }
    return result;
}

double final_prob(const std::vector<double>& recalls) {
  double result = 1.0;
  for (auto val : recalls) {
    result *= 1 - val;
  }
  return 1 - result;
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
  int hash_funcs_per_table = conf.hash_table_params[0].k;
  int num_hash_funcs = num_tables * hash_funcs_per_table;
  int hash_table_width = conf.hash_table_width;
  float bucket_width = conf.hash_table_params[0].bucket_width;
  int probe_tables = num_tables;
  if (argc >= 4) {
    probe_tables = atoi(argv[3]);
  }
  // Probablity of spontaneous collision due to low hash table width.
  double spontaneous_coll_prob = std::pow(2, -hash_table_width);

  auto distances_raw = read_data<VectorType>(conf.distance_filename);
  auto query_hash_raw = read_data<VectorType>(conf.query_hash_filename);
  MatrixType distances(conf.num_queries, conf.num_points);
  MatrixType query_hash(conf.num_queries, num_hash_funcs);
  for (int row = 0; row < conf.num_queries; ++row) {
    distances.row(row) = distances_raw[row];
    query_hash.row(row) = query_hash_raw[row];
  } 
  
  auto gaussian_cdf = read_data<VectorType>(conf.eigen_filename)[0];
  auto probing_sequence = read_data<DenseVector<int>>(conf.probing_sequence_file);
  auto probing_iter = probing_sequence.begin();

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

      if (table >= probe_tables) {
        // this candidate is not found within our limit table_num
        continue;
      }
      candidates[qid].push_back(id);
    }
  }

  std::vector<std::vector<double>> recall_ps(conf.num_queries);
  for (int qid = 0; qid < conf.num_queries; ++qid) {
    if (qid > fast_stop) break;

    int probe_size = (*probing_iter)[0];
    MatrixType probes(probe_size, hash_funcs_per_table);  // table id is at last of each row
    std::vector<int> tables(probe_size);
    for (int row = 0; row < probe_size; ++row) {
      auto& line = *(probing_iter + row + 1);
      probes.row(row) = line(Eigen::seq(0, hash_funcs_per_table - 1)).template cast<float>();
      tables[row] = line(hash_funcs_per_table);
    }
    probing_iter += probe_size + 1;

    for (int id : candidates[qid]) {
      std::vector<double> recalls(num_tables, 0);
      // std of data distribution
      double scale = distances(qid, id) / bucket_width;

      for (int bucket = 0; bucket < probe_size; ++bucket) {
        auto query_center = query_hash(qid,
            Eigen::seqN(tables[bucket] * hash_funcs_per_table, hash_funcs_per_table));
        VectorType lower_vec = probes.row(bucket) - query_center;
        VectorType upper_vec = lower_vec + VectorType::Ones(hash_funcs_per_table);
        // normalize to standard gaussian
        lower_vec = lower_vec / scale;
        upper_vec = upper_vec / scale;
        recalls[tables[bucket]] += bucket_prob(gaussian_cdf, lower_vec, upper_vec)
            + spontaneous_coll_prob;
      }
      double final_recall = final_prob(recalls);
      /*if (distances(qid, id) > 1) {
        using namespace ONIAK;
        std::cout << "far: " << distances(qid, id) << "\t" << id <<  std::endl;
        std::cout << recalls << std::endl;
        break;
      } */
      recall_ps[qid].push_back(final_recall);
    }
  }

  std::ofstream fout(conf.recall_p_filename);
  for (auto& recall : recall_ps) {
    write_data(fout, recall);
  }

  return 0;
}