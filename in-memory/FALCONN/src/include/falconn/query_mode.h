#ifndef __QUERY_MODE_H__
#define __QUERY_MODE_H__

#include <fstream>
#include <filesystem>
#include <vector>
#include <type_traits>
#include "falconn_global.h"
#include "lsh_nn_table.h"
#include "fileio.h"
#include <Timer.hpp>
#include <StringUtils.hpp>
#include <AnnResultWriter.hpp>

#include "oniak/misc.hpp"
#include "oniak/recall.hpp"

namespace falconn {

template <typename... Targs>
unsigned char compact_bool_recur(unsigned char compact, bool cur, Targs... Fargs) {
  compact <<= 1;
  compact |= static_cast<char>(cur);
  if constexpr (sizeof...(Targs) > 0) {
    compact = compact_bool_recur(compact, Fargs...);
  }
  return compact;
}

// Compacts its arguments into a single byte.
template <typename... Targs>
unsigned char compact_bool(Targs... Fargs) {
  unsigned char compact = 0;
  constexpr size_t sz = sizeof...(Targs);
  static_assert(sz <= sizeof(unsigned char) * 8);
  compact = compact_bool_recur(compact, Fargs...);
  compact <<= (sizeof(unsigned char) * 8 - sz);
  return compact;
}

// File format:
// First byte is shows which fields are printed.
// id - table - bucket - distance - ? - ? - ? - ?
//
// Then, 4 bytes give how many candidates are dumped.
//
// For each candidate, id, table, buckets are ints (4 bytes),
// distance is float (4 bytes).
void print_candidates(const std::vector<FalconnCandidateType>& candidates,
                      std::string filename, const CandidatePrintingMode& mode) {
  std::ofstream fout(filename, std::ios::binary);
  // The output must be valid.
  if (!fout) {
    std::cout << filename << "cannot be opened! Aborting." << std::endl;
    exit(1);
  }
  unsigned char flags = compact_bool(mode.print_id, mode.print_table, mode.print_bucket_order,
                                     mode.print_distance);
  fout.write(reinterpret_cast<char*>(&flags), sizeof(flags));
  int num_candidates = static_cast<int>(candidates.size());
  fout.write(reinterpret_cast<char*>(&num_candidates), sizeof(int));

  for (const FalconnCandidateType& candi : candidates) {
    if (mode.print_id) {
      int candidate_id = static_cast<int>(candi.id);
      fout.write(reinterpret_cast<char*>(&candidate_id), sizeof(int));
    }
    if (mode.print_table) {
      int table = static_cast<int>(candi.table);
      fout.write(reinterpret_cast<char*>(&table), sizeof(int));
    }
    if (mode.print_bucket_order) {
      int bucket_order = static_cast<int>(candi.bucket_order);
      fout.write(reinterpret_cast<char*>(&bucket_order), sizeof(int));
    }
    if (mode.print_distance) {
      int q2d_distance = static_cast<float>(candi.q2d_distance);
      fout.write(reinterpret_cast<char*>(&q2d_distance), sizeof(float));
    }
  }
}

void print_hashed_queries(const LSHConstructionParameters& params) {
  falconn::PointSet train;
  // empty training set
  auto table = construct_table<PointType>(train, params);
  falconn::QueryIterator<falconn::FalconnQueryType> test =
    typename falconn::QueryIterator<falconn::FalconnQueryType>(params.query_filename);
  NPP_ENFORCE(test.size() == params.num_queries);

  int qn = params.num_queries;
  // double candidate_sum = 0., recall_sum = 0., query_time_sum = 0., query_raw_sum = 0.;
  auto query_object_ori = table->construct_query_object(
    params.hash_table_params[0].l * params.probes_per_table, -1, 1, 3);
  auto* query_object = query_object_ori.get();

  auto query_iter = test.begin();
  std::ofstream fout(params.query_hash_filename);
  for (int i = 0; i < qn; i++, ++query_iter) {
    auto hashed_query = query_object->get_transformed_vector(*query_iter);
    write_data<VectorType>(fout, hashed_query);
  }
  std::cout << "Hashed query values dumped to " << params.query_hash_filename << std::endl;
  return;
}

void print_probing_sequence(const LSHConstructionParameters& params) {
  falconn::PointSet train;
  // empty training set
  auto table = construct_table<PointType>(train, params);
  falconn::QueryIterator<falconn::FalconnQueryType> test =
    typename falconn::QueryIterator<falconn::FalconnQueryType>(params.query_filename);
  NPP_ENFORCE(test.size() == params.num_queries);

  int qn = params.num_queries;
  auto query_object_ori = table->construct_query_object(
    params.hash_table_params[0].l * params.probes_per_table, -1, 1, 3);
  auto* query_object = query_object_ori.get();

  auto query_iter = test.begin();
  std::ofstream fout(params.probing_sequence_file);
  for (int i = 0; i < qn; i++, ++query_iter) {
    auto probing_sequence = query_object->get_probing_sequence(*query_iter);
    // table id is at the end of each vector.
    std::vector<int> size_vec = { static_cast<int>(probing_sequence.size()) };
    write_int_data(fout, size_vec);
    for (auto& probe : probing_sequence) {
      write_int_data(fout, probe);
    }
  }
  std::cout << "Probing sequence dumped to " << params.probing_sequence_file << std::endl;
  return;
}

void print_precomputed_mp_sequence(const LSHConstructionParameters& params) {
  falconn::PointSet train;
  // empty training set
  auto table = construct_table<PointType>(train, params);
  auto query_object_ori = table->construct_query_object(
    params.hash_table_params[0].l * params.probes_per_table, -1, 1, 3);
  auto* query_object = query_object_ori.get();

  auto mp_pointer = query_object->internal_query()->lsh_query().multiprobe();
  auto precomputed_sequence = mp_pointer->precomputed();
  std::ofstream fout(params.probing_sequence_file);
  for (auto row : precomputed_sequence) {
    write_int_data(fout, row);
  }
  std::cout << "Precomputed probing sequence dumped to " << params.probing_sequence_file << std::endl;
  return;
}

void print_hash_function(const LSHConstructionParameters& params) {
  falconn::PointSet train;
  // empty training set
  auto table = construct_table<PointType>(train, params);
  auto query_object_ori = table->construct_query_object(
    params.hash_table_params[0].l * params.probes_per_table, -1, 1, 3);
  auto* query_object = query_object_ori.get();

  auto hash_func = query_object->internal_query()->lsh_query().hash_function();
  hash_func.dump(params.hash_func_filename);
  std::cout << "Hash function realizations dumped to " << params.hash_func_filename << std::endl;
  return;
}

void qdde_ground_truth(const LSHConstructionParameters& params) {
  // Raw distance data is input via histogram_filename.
  std::ifstream fin(params.histogram_filename);
  std::cout << "Reading distance matrix from " << params.histogram_filename << std::endl;
  MatrixType distance = read_one_matrix<CoordinateType>(fin, params.num_queries, params.num_points);
  const FalconnRange& range = params.bins_vector[0];
  std::vector<int> selected_queries;
  auto kde_gt = ONIAK::QDDE_gt(distance, range, selected_queries);

  std::string gnd_filename = std::vformat(params.gnd_filename, std::make_format_args(
    params.bins_vector[0].start, params.bins_vector[0].end, params.bins_vector[0].count));
  std::string rowid_filename = std::vformat(params.rowid_filename, std::make_format_args(
    params.bins_vector[0].start, params.bins_vector[0].end, params.bins_vector[0].count));
  std::ofstream fout(gnd_filename);
  for (auto& vec : kde_gt) {
    write_double_data(fout, vec);
  }
  fout.close();
  fout.open(rowid_filename);
  write_int_data(fout, selected_queries);
}

void cande_main(const LSHConstructionParameters& params, const FalconnQueryIterator& queries,
                FalconnQueryWrapper* query_object) {
  assert(params.bins_vector.size() == 1);
  assert(params.hash_table_params.size() == 1 && params.hash_table_params[0].l < 200);

  std::vector<KeyType> knn_result;
  std::vector<CoordinateType> val_result;
  std::vector<std::vector<double>> maes, mres, kde_gt;
  std::vector<int> selected_queries;

  std::string gnd_filename = params.gnd_filename;
  std::string rowid_filename = params.rowid_filename;

  if (params.cande_task == CanDETask::kQDDE) {
    gnd_filename = std::vformat(params.gnd_filename, std::make_format_args(
      params.bins_vector[0].start, params.bins_vector[0].end, params.bins_vector[0].count));
    rowid_filename = std::vformat(params.rowid_filename, std::make_format_args(
      params.bins_vector[0].start, params.bins_vector[0].end, params.bins_vector[0].count));

    if (!std::filesystem::exists(gnd_filename) || !std::filesystem::exists(rowid_filename)) {
      qdde_ground_truth(params);
    }
  }
  if (params.cande_task == CanDETask::kKDE || params.cande_task == CanDETask::kQDDE) {
    kde_gt = read_data<std::vector<double>>(gnd_filename);
    selected_queries = read_data<std::vector<int>>(rowid_filename)[0];
  }

  for (int num_neighbors : params.num_neighbors_list) {
    std::ifstream gt_in(params.gnd_filename);
    IntegerMatrix ground_idx;
    if (params.cande_task == CanDETask::kRecall) {
      ground_idx = falconn::read_ground_truth(gt_in, num_neighbors, params.num_queries);
    }

    for (auto algo_type : params.cande_algos) {
      assert(algo_type != CanDEAlgoType::kUnknown);
      std::vector<std::vector<KeyType>> all_keys;
      std::vector<std::vector<CoordinateType>> all_values;
      all_keys.reserve(params.num_queries);
      all_values.reserve(params.num_queries);

      for (auto query_iter = queries.begin(); query_iter != queries.end(); ++query_iter) {
        if (algo_type == CanDEAlgoType::kHash) {
          if (params.cande_task == CanDETask::kKDE) {
            query_object->knn_and_kde_infer(*query_iter, query_iter.qid(), num_neighbors, &knn_result,
                                            &val_result);
          } else {
            query_object->knn_and_qdde_infer(*query_iter, query_iter.qid(), num_neighbors,
                                             &knn_result, &val_result, params.cande_task);
          }
        } else if (algo_type == CanDEAlgoType::kPrecompute) {
          query_object->knn_and_precomputed(*query_iter, query_iter.qid(), num_neighbors, &knn_result,
                                            &val_result, params.cande_task);
        } else if (algo_type == CanDEAlgoType::kAssociative) {
          query_object->knn_infer_associative(*query_iter, query_iter.qid(), num_neighbors, &knn_result,
                                              &val_result, params.cande_task);
        } else if (algo_type == CanDEAlgoType::kCPAdjusted) {
          query_object->knn_and_cande_cp_adjusted(*query_iter, query_iter.qid(), num_neighbors, &knn_result,
                                                  &val_result, params.cande_task);
        } else if (algo_type == CanDEAlgoType::kResample) {
          query_object->knn_and_cande_resample(*query_iter, query_iter.qid(), num_neighbors, &knn_result,
                                               &val_result, params.cande_task);
        } else {
          throw std::runtime_error("Unknown CanDE algorithm type");
        }
        std::sort(knn_result.begin(), knn_result.end());
        all_keys.push_back(std::move(knn_result));
        all_values.push_back(std::move(val_result));
      }

      if (params.compute_gound_truth) {
        std::vector<double> cur_mae, cur_mre;
        if (params.cande_task == CanDETask::kRecall) {
          std::vector<double> recalls = ONIAK::recall_all_q(all_keys, ground_idx, num_neighbors);
          std::vector<double> recalls_est = ONIAK::get_column<float, double>(all_values, 0);
          cur_mae.push_back(ONIAK::mae(recalls_est, recalls));
          cur_mre.push_back(ONIAK::mre(recalls_est, recalls, 0.02));
        } else {  // KDE or QDDE
          // transposed values are gamma (bin) by query
          auto transposed_values = ONIAK::transpose(all_values);
          cur_mae = ONIAK::mae_all(transposed_values, kde_gt, selected_queries);
          cur_mre = ONIAK::mre_all(transposed_values, kde_gt, selected_queries);
        }
        maes.push_back(cur_mae);
        mres.push_back(cur_mre);
        std::cout << std::format("CanDE-{} finished using algo {} at k={} for {} queries. MAE={}, MRE={}.",
                                 cande_task_name(params.cande_task), cande_algo_name(algo_type),
                                 num_neighbors, params.num_queries, vec_to_string(cur_mae), vec_to_string(cur_mre))
          << std::endl;
      }

      // Time to save all results.
      auto query_statistics = query_object->get_query_statistics();
      std::string result_filename =
        std::vformat(params.result_filename, std::make_format_args(cande_algo_name(algo_type), num_neighbors));
      std::ofstream fout(result_filename);
      query_statistics.dump(fout);
      // knn results will not be repeated for a different algorithm if already exists
      std::string knn_filename =
        std::vformat(params.knn_filename, std::make_format_args(num_neighbors));
      if (!std::filesystem::exists(knn_filename)) {
        std::ofstream knn_out(knn_filename);
        for (auto& vec : all_keys) {
          write_int_data(knn_out, vec);
        }
      }
      std::string val_filename =
        std::vformat(params.kde_filename, std::make_format_args(cande_algo_name(algo_type), num_neighbors));
      std::ofstream val_out(val_filename);
      for (auto& vec : all_values) {
        write_data(val_out, vec);
      }
    }
  }
  if (params.compute_gound_truth) {
    // Dump accuracy results of all experiments
    std::ofstream accuracy_out(params.accuracy_filename);
    if (!accuracy_out) {
      std::filesystem::create_directories(
        std::filesystem::path(params.accuracy_filename).parent_path());
      accuracy_out.close();
      accuracy_out.open(params.accuracy_filename);
    }
    accuracy_out << "Algorithm\t#NNs\tMRE\tMAE" << std::endl;
    auto mae_iter = maes.begin();
    auto mre_iter = mres.begin();
    for (int num_neighbors : params.num_neighbors_list) {
      for (auto algo_type : params.cande_algos) {
        accuracy_out << std::format("{}\t{}\t{}\t{}",
                                    cande_algo_name(algo_type), num_neighbors, vec_to_string(*mre_iter),
                                    vec_to_string(*mae_iter))
          << std::endl;
        ++mae_iter;
        ++mre_iter;
      }
    }

    // data is dumped in dvecs double format
    std::ofstream accuracy_binary_out(params.accuracy_binary_filename);
    if (!accuracy_binary_out) {
      std::filesystem::create_directories(
        std::filesystem::path(params.accuracy_binary_filename).parent_path());
      accuracy_binary_out.close();
      accuracy_binary_out.open(params.accuracy_binary_filename);
    }
    for (auto& mae : maes) {
      write_data(accuracy_binary_out, mae);
    }
    for (auto& mre : mres) {
      write_data(accuracy_binary_out, mre);
    }
  }
}

void falconn_query_mode() {
  // The following modes do not need a full hash index.
  const LSHConstructionParameters& params = falconn_config;
  assert(params.query_mode != FalconnQueryMode::Unknown);
  if (params.query_mode == FalconnQueryMode::PrintHashedQueries) {
    print_hashed_queries(params);
    return;
  } else if (params.query_mode == FalconnQueryMode::PrintProbingSequence) {
    print_probing_sequence(params);
    return;
  } else if (params.query_mode == FalconnQueryMode::PrintPrecomputedMPSequence) {
    print_precomputed_mp_sequence(params);
    return;
  } else if (params.query_mode == FalconnQueryMode::PrintHashFunction) {
    print_hash_function(params);
    return;
  } else if (params.query_mode == FalconnQueryMode::QDDEGroundTruth) {
    qdde_ground_truth(params);
    return;
  }

  auto train = falconn::read_data<PointType>(params.data_filename);
  auto table = construct_table<PointType>(train, params);
  auto test = FalconnQueryIterator(params.query_filename);
  if (params.query_mode != FalconnQueryMode::CandidateNum) {
    NPP_ENFORCE(test.size() == params.num_queries);
  }
  auto query_object_ori = table->construct_query_object(
    params.hash_table_params[0].l * params.probes_per_table, -1, 1, 3);
  auto* query_object = query_object_ori.get();
  HighResolutionTimer timer;

  int qn = params.num_queries, K = params.num_neighbors;

  if (params.query_mode == FalconnQueryMode::PrintCandidates ||
      params.query_mode == FalconnQueryMode::PrintDuplicateCandidates) {
    auto query_iter = test.begin();
    for (int i = 0; i < qn; i++) {
      std::vector<FalconnCandidateType> res;
      timer.restart();
      query_object->reset_query_statistics();
      if (params.query_mode == FalconnQueryMode::PrintCandidates) {
        query_object->get_unique_candidates(*query_iter, &res);
      } else {
        query_object->get_candidates_with_duplicates(*query_iter, &res);
      }

      std::cout << "Number of candidates for query " << i << ": " << res.size() << std::endl;
      std::cout << "returned in time (us): " << timer.elapsed() << std::endl;

      std::string result_file = params.candidate_filename + "_q_" + std::to_string(i) + ".bin";
      print_candidates(res, result_file, params.printing_mode);
      ++query_iter;
    }
  } else if (params.query_mode == FalconnQueryMode::CandidateNum) {
    auto query_iter = test.begin();
    std::ofstream fout(params.candidate_filename);
    for (int i = 0; i < qn; i++) {
      std::vector<int> candidateNum;
      candidateNum = query_object->get_knn_candidate(*query_iter, params.num_neighbors);
      write_int_data(fout, candidateNum);
      ++query_iter;
    }
    fout.close();
  } else if (params.query_mode == FalconnQueryMode::KNNWithCanDE) {
    cande_main(params, test, query_object);
  } else if (params.query_mode == FalconnQueryMode::PrintTimeMeasurements ||
             params.query_mode == FalconnQueryMode::MeasureKNNTime) {
    auto query_iter = test.begin();

    for (int i = 0; i < qn; i++) {
      if (params.query_mode == FalconnQueryMode::PrintTimeMeasurements) {
        std::vector<FalconnCandidateType> res;
        query_object->get_unique_candidates(*query_iter, &res);
      } else if (params.query_mode == FalconnQueryMode::MeasureKNNTime) {
        std::vector<KeyType> res;
        query_object->find_k_nearest_neighbors(*query_iter, params.num_neighbors, &res);
      } else {
        assert(false);
      }
      ++query_iter;
    }
    auto query_statistics = query_object->get_query_statistics();
    std::ofstream fout(params.result_filename);
    query_statistics.dump(fout);
    std::cout << "Time measurements dumped to: " << params.result_filename << std::endl;
  } else if (params.query_mode == FalconnQueryMode::KNNWithTime) {
    std::ifstream gt_in(params.gnd_filename);
    auto ground_idx = falconn::read_ground_truth(gt_in, K, qn);

    // TODO
    assert(false);
  } else if (params.query_mode == FalconnQueryMode::KNNRecall) {
    std::ifstream gt_in(params.gnd_filename);
    std::ifstream dist_in(params.distance_filename);
    IntegerMatrix ground_idx;
    MatrixType ground_distance;
    if (params.query_variant == NNQueryVariant::TopK) {
      ground_idx = falconn::read_ground_truth(gt_in, K, qn);
    } else if (params.query_variant == NNQueryVariant::RadiusR) {
      ground_distance = falconn::read_one_matrix<CoordinateType>(dist_in, qn, params.num_points);
    }

    std::string summary_path = std::vformat(
      params.summary_path,
      std::make_format_args(params.hash_table_params[0].l, params.hash_table_params[0].k,
                            params.hash_table_params[0].bucket_width));
    
    CppFileHelper summary_file(summary_path);
    double candidate_sum = 0., recall_sum = 0., query_time_sum = 0., query_raw_sum = 0.;
    auto query_iter = test.begin();
    int num_valid_queries = 0;
    std::vector<double> recalls(qn);
    // std::vector<std::vector<int>> res_result;
    for (int i = 0; i < qn; i++) {
      std::vector<FalconnCandidateType> candidates;
      timer.restart();
      query_object->reset_query_statistics();
      query_object->get_unique_candidates(*query_iter, &candidates);
      auto query_time = timer.elapsed();

      std::vector<int> ground_truth_rr;
      if (params.query_variant == NNQueryVariant::RadiusR) {
        assert(params.radius_R.size() > 0);
        for (int nid = 0; nid < params.num_points; ++nid) {
          if (ground_distance(i, nid) < params.radius_R[0]) {
            ground_truth_rr.push_back(nid);
          }
        }
        std::sort(ground_truth_rr.begin(), ground_truth_rr.end());
      }

      const std::vector<int>& ground_truth =
        (params.query_variant == NNQueryVariant::TopK) ? ground_idx[i] : ground_truth_rr;
      if (ground_truth.size() == 0) {
        ++query_iter;
        continue;
      }
      std::vector<int> res;
      res.reserve(candidates.size());
      std::transform(candidates.begin(), candidates.end(), std::back_inserter(res),
                     [](const FalconnCandidateType& cand) { return cand.id; });
      std::sort(res.begin(), res.end());
      //res_result.push_back(res);
      std::vector<int> v_intersection;
      std::set_intersection(ground_truth.begin(),
                            ground_truth.end(), res.begin(), res.end(), std::back_inserter(v_intersection));
      double recall = (float) v_intersection.size() / ground_truth.size();
      recalls[i] = recall;
      auto statistics = query_object->get_query_statistics();
      double candi_num = statistics.average_num_unique_candidates;
      candidate_sum += candi_num;
      recall_sum += recall;
      query_raw_sum += query_time;
      query_time_sum += statistics.average_total_query_time;

      ++query_iter;
      ++num_valid_queries;
    }

    auto recall_out = open_file_create(params.kde_filename);
    write_data(recall_out, recalls);
    
    // std::ofstream recall_test(params.accuracy_filename);
    // for (auto &vec : res_result)
    // {
    //   write_data(recall_test, vec);
    // }

    summary_file.print_one_line("l", "m", "w", "recall", "candidate ratio", "query time");
    summary_file.print_one_line(params.hash_table_params[0].l, params.hash_table_params[0].k,
                                params.hash_table_params[0].bucket_width, recall_sum / num_valid_queries,
                                candidate_sum / ((double) params.num_points * (double) num_valid_queries),
                                query_time_sum / num_valid_queries);
  }
}

} // namespace falconn

#endif