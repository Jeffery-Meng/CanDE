#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "stop_watch.hpp"
#include "simulation/data_generator.h"
#include "simulation/config.hpp"

#include "core/polytope_hash2.h"
#include "core/multiprobe_cp.h"

using namespace ONIAK;
using namespace std;
using namespace falconn;
using namespace falconn::core;

using Multiprobe = MultiProbeCP<CrossPolytopeHash2<>>;
using BucketVector = std::vector<std::vector<HashType>>;
using MultiprobeVector = std::vector<typename CrossPolytopeHash2<>::MultiprobeType>;
using HashTran = typename CrossPolytopeHash2<>::HashTransformation;
//buckets : # data * # hash functions 

std::vector<double> evaluate_lsh(std::vector<Multiprobe>& mps, std::vector<HashTran>& hash_tran,
        const DMatrix& input1s, const DMatrix& input2s, const DMatrix& input3s, const DMatrix& input4s,
        double& total_time, int T) {
    StopWatch timer(false);
    std::vector<double> collisions(T, 0);
    size_t num_hashes = mps.size();
    size_t num_experiments = input1s.rows();

    timer.start();
    MultiprobeVector mp_vec;
    for (int exp = 0; exp < num_experiments; ++exp) {
      DRowVector v1 = input1s(exp, Eigen::all);
      DRowVector v2 = input2s(exp, Eigen::all);
      DRowVector v3 = input3s(exp, Eigen::all);
      DRowVector v4 = input4s(exp, Eigen::all);
      for (int hsh = 0; hsh < num_hashes; ++hsh) {
        hash_tran[hsh].apply(std::make_pair(v1, v3), &mp_vec, T);
        mps[hsh].setup_probing(std::move(mp_vec), hash_tran[hsh].get_l());
        HashType data_pb;
        int_fast32_t data_table;
        mps[hsh].get_next_probe(&data_pb, &data_table);

        hash_tran[hsh].apply(std::make_pair(v2, v4), &mp_vec, T);
        mps[hsh].setup_probing(std::move(mp_vec), hash_tran[hsh].get_l() * T);
          for (int t = 0; t < T; ++t) {
            HashType pb;
            int_fast32_t table;
            mps[hsh].get_next_probe(&pb, &table);
            // table is always 0.
            if (pb == data_pb) {
              for (; t < T; ++t) {
                collisions[t] += 1;
              }
            }
          }
      }
    }
    timer.stop();
    total_time = timer.peek() / num_hashes / num_experiments * 1e6;
    for (auto& val: collisions) {
      val /= num_hashes * num_experiments;
    }
    return collisions;
}

int main(int argc, char * argv[]) {
  assert(argc > 1);
    size_t seed;
    // seed is unchanged if configuration remains the same.
    KroneckerExperimentConfig config = read_config(argv[1], &seed);
    std::mt19937 rng(seed);
    std::ofstream output_stream(config.output_path);
    DRowVector candidates(config.probe_buckets);
    std::iota(candidates.begin(), candidates.end(), 1);
    output_stream << "Cosine\ttime (us)\t" << candidates << endl;

    std::vector<std::unique_ptr<CrossPolytopeHash2<>>> lshes;
    std::vector<Multiprobe> mps;
    std::vector<HashTran> hash_tran;
    lshes.reserve(config.num_hashes);
    for (int hsh = 0; hsh < config.num_hashes; ++hsh) {
      std::unique_ptr<CrossPolytopeHash2<>> cp_hash = 
          std::make_unique<CrossPolytopeHash2<>>(config.dim, config.k, /*L*/ 1, rng(), /*hash width*/ 20);
      mps.emplace_back(*cp_hash, config.probe_buckets);
      hash_tran.emplace_back(*cp_hash);
      lshes.push_back(std::move(cp_hash));
    }

    for (DType cosine : config.cosines) {
        cout << "Running for cosine: " << cosine << endl;
        DMatrix input1s(config.num_experiments, config.dim);
        DMatrix input2s(config.num_experiments, config.dim);
        DMatrix input3s(config.num_experiments, config.dim);
        DMatrix input4s(config.num_experiments, config.dim);
        for (int exp = 0; exp < config.num_experiments; ++exp) {
            auto [vector1, vector2] = TwoRandomVectors(config.dim, cosine, rng);
            auto [vector3, vector4] = TwoRandomVectors(config.dim, cosine, rng);
            input1s(exp, Eigen::all) = vector1;
            input2s(exp, Eigen::all) = vector2;
            input3s(exp, Eigen::all) = vector3;
            input4s(exp, Eigen::all) = vector4;
        }

        double kronecker_time;
        auto kronecker_collisions = evaluate_lsh(mps, hash_tran, input1s, input2s, input3s, input4s,
                                   kronecker_time, config.probe_buckets);
        print_one_line(std::cout, cosine * cosine, kronecker_time, kronecker_collisions);
        print_one_line(output_stream, cosine * cosine, kronecker_time, kronecker_collisions);
    }
}