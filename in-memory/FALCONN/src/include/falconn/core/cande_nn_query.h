#ifndef __CANDE_NN_QUERY_H__
#define __CANDE_NN_QUERY_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "../oniak/hash_table.hpp"
#include "../oniak/misc.hpp"
#include "../fileio.h"
#include "../oniak/real_array.hpp"
#include "../falconn_global.h"
#include "heap.h"
#include "data_storage.h"
#include "nn_query.h"
#include <limits>

using std::exception;
using std::runtime_error;
using std::string;

namespace falconn
{
  namespace core
  {

    template <typename LSHTableQuery, typename LSHTablePointType,
              typename LSHTableKeyType, typename ComparisonPointType,
              typename DistanceType, typename DistanceFunction,
              typename DataStorage>
    class CanDENearestNeighborQuery
    {
    public:
      typedef FalconnQueryType QueryType;
      using dist_table_pair = std::pair<float, uint32_t>;

      CanDENearestNeighborQuery(LSHTableQuery *table_query,
                                const DataStorage &data_storage)
      try : table_query_(table_query), data_storage_(data_storage),
          mp_recalls_(falconn_config.mp_prob_filename),
          mp_recalls1_(mp_recalls_),
          mp_recalls_lm1_(mp_recalls_),
          table_bin_ht_(*falconn_config.seeding_sequence, falconn_config.cande_table_size),
          knn_tables_(*falconn_config.seeding_sequence, 16384),
          hash_func_(falconn_config.seed, 0xffffffff),
          k_(0), num_inserted_(0),
          bayesian_prior_(falconn_config.bayesian_prior)
      {
        // raise single-table Multiprobe recalls to that of L tables
        int num_tables = falconn_config.hash_table_params[0].l;
        float bucket_width = falconn_config.hash_table_params[0].bucket_width;

        FalconnRange &range = falconn_config.bins_vector[0];
        cps_bin_per_table_.resize(range.num_bins());

        // normalize distance by bucket width
        mp_recalls_.start() *= bucket_width;
        mp_recalls_.step() *= bucket_width;
        mp_recalls1_.start() *= bucket_width;
        mp_recalls1_.step() *= bucket_width;
        mp_recalls_lm1_.start() *= bucket_width;
        mp_recalls_lm1_.step() *= bucket_width;
        // recalls values read from files are for one table
        // Change to overall recall of L tables
        for (auto &val : mp_recalls_.array())
        {
          val = 1.0 - std::pow(1.0 - val, num_tables);
        }
        for (auto &val : mp_recalls_lm1_.array())
        {
          val = 1.0 - std::pow(1.0 - val, num_tables - 1);
        }
      }
      catch (std::filesystem::filesystem_error &e)
      {
        std::cout << "Cannot open MP recall file. Aborting" << std::endl;
        exit(1);
      }

      KeyType find_nearest_neighbor(const QueryType &q,
                                    const QueryType &q_comp,
                                    int_fast64_t num_probes,
                                    int_fast64_t max_num_candidates)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);
        auto distance_start_time = std::chrono::high_resolution_clock::now();

        // TODO: use nullptr for pointer types
        LSHTableKeyType best_key = -1;

        if (candidates_.size() > 0)
        {
          typename DataStorage::SubsequenceIterator iter =
              data_storage_.get_subsequence(candidates_);

          best_key = candidates_[0];
          DistanceType best_distance = dst_(q_comp, iter.get_point());
          ++iter;

          // printf("%d %f\n", candidates_[0], best_distance);

          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            // printf("%d %f\n", iter.get_key(), cur_distance);
            if (cur_distance < best_distance)
            {
              best_distance = cur_distance;
              best_key = iter.get_key();
              // printf("  is new best\n");
            }
            ++iter;
          }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();

        return best_key;
      }

      void find_k_nearest_neighbors(const QueryType &q,
                                    const QueryType &q_comp,
                                    int_fast64_t k, int_fast64_t num_probes,
                                    int_fast64_t max_num_candidates,
                                    std::vector<LSHTableKeyType> *result)
      {
        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);

        heap_.reset();
        heap_.resize(k);

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);

        int_fast64_t initially_inserted = 0;
        for (; initially_inserted < k; ++initially_inserted)
        {
          if (iter.is_valid())
          {
            heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
            ++iter;
          }
          else
          {
            break;
          }
        }

        if (initially_inserted >= k)
        {
          heap_.heapify();
          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            if (cur_distance < -heap_.min_key())
            {
              heap_.replace_top(-cur_distance, iter.get_key());
            }
            ++iter;
          }
        }

        res.resize(initially_inserted);
        std::sort(heap_.get_data().begin(),
                  heap_.get_data().begin() + initially_inserted);
        for (int_fast64_t ii = 0; ii < initially_inserted; ++ii)
        {
          res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      void knn_and_kde_infer(const QueryType &q, int qid,
                             int_fast64_t k, int_fast64_t num_probes,
                             int_fast64_t max_num_candidates,
                             std::vector<LSHTableKeyType> *result,
                             std::vector<CoordinateType> *kde_result)
      {
        if (result == nullptr || kde_result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }
        auto start_time = std::chrono::high_resolution_clock::now();

        FalconnRange &range = falconn_config.bins_vector[0];
        int num_tables = falconn_config.hash_table_params[0].l;
        int num_bins = range.num_bins();
        table_bin_ht_.clear();
        std::vector<int> candidate_num_per_table;

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        // Step 1: get duplicate candidates
        table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                                     &candidates_, &candidate_num_per_table);
        k_ = k;
        num_inserted_ = 0;
        heap_.reset();
        heap_.resize(k);

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);

        std::vector<double> recalls(num_bins, 0.0);
        std::vector<float> distances;
        std::vector<int> uniq_cands_per_bin(num_bins, 0);
        // candidates by table
        std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));
        int gamma_num = falconn_config.gamma.size();
        std::vector<float> gammas(gamma_num);
        std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
                       [](float x)
                       { return x * x * 2.0; });
        std::vector<std::vector<float>> kde_sums(gamma_num, std::vector<float>(num_bins, 0.0));
        // # candidates unique to a given table
        auto utcands_per_bin = tcands_per_bin;

        auto distance_start_time = std::chrono::high_resolution_clock::now();
        // Step 2a: compute all distances
        std::string distance_filename = std::vformat(falconn_config.distance_filename,
                                                     std::make_format_args("dup", qid));
        if (falconn_config.load_distance && std::filesystem::exists(distance_filename))
        {
          distances = read_data<std::vector<float>>(distance_filename)[0];
        }
        else
        {
          distances.reserve(candidates_.size());
          while (iter.is_valid())
          {
            DistanceType distance = dst_(q, iter.get_point());
            distances.push_back(distance);
            ++iter;
          }
        }
        if (falconn_config.load_distance && !std::filesystem::exists(distance_filename))
        {
          std::ofstream fout(distance_filename);
          if (!fout)
          {
            std::filesystem::create_directories(
                std::filesystem::path(distance_filename).parent_path());
            fout.close();
            fout.open(distance_filename);
          }
          write_data(fout, distances);
        }

        int table = 0, count = 0;
        // Step 2b: deduplicate using hash tables, and update inference counters
        for (auto key : candidates_)
        {
          ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
          float distance = distances[count];
          int bin = range.bin_translate(distance);
          while (count >= candidate_num_per_table[table])
            ++table;
          auto &table_find = table_bin_ht_.find_or_insert(key, status);
          // ONIAK::TableBinPair table_bin = {2,2};
          if (status == ONIAK::HashInsertionStatus::kNewlyInserted)
          {
            // first occurrence of candidate
            // table_bin.bin = bin;
            table_find = table;
            ++tcands_per_bin[bin][table];
            ++utcands_per_bin[bin][table];
            ++uniq_cands_per_bin[bin];
            for (int gammaid = 0; gammaid < gamma_num; ++gammaid)
            {
              kde_sums[gammaid][bin] += ONIAK::kdev2(distance, gammas[gammaid]);
            }
            insert_heap(distance, key);
          }
          else if (status == ONIAK::HashInsertionStatus::kAlreadyExists)
          {
            ++tcands_per_bin[bin][table];
            if (table_find != 255)
            {
              --utcands_per_bin[bin][table_find];
              table_find = 255;
            }
          }
          else
          { // if hash table is full, only check for knn
            if (num_inserted_ < 0)
            {
              insert_heap(distance, key);
            } // otherwise the result may contain duplicates.
          }
          ++count;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        stats_.average_distance_time += elapsed_distance.count();

        // step 3: return top Knn results
        dump_knn(res);

        // step 4: infer recalls of each bin
        for (int bin = 0; bin < num_bins; ++bin)
        {
          std::vector<double> rhos(num_tables, 0.0);
          for (int table = 0; table < num_tables; ++table)
          {
            rhos[table] = bayesian_inference(tcands_per_bin[bin][table],
                                             utcands_per_bin[bin][table],
                                             uniq_cands_per_bin[bin],
                                             range.mid_point(bin));
          }
          double rho = ONIAK::final_prob(rhos);
          recalls[bin] = rho;
        }

        // step 5: infer KDE
        kde_result->assign(gamma_num, 0.0);
        for (int gammaid = 0; gammaid < gamma_num; ++gammaid)
        {
          for (int bin = 0; bin < num_bins; ++bin)
          {
            if (recalls[bin] > 1e-4)
            {
              kde_result->operator[](gammaid) += kde_sums[gammaid][bin] / recalls[bin];
            }
          }
        }

        auto sketches_end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_sketches =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                sketches_end_time - end_time);
        stats_.average_sketches_time += elapsed_sketches.count();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(sketches_end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
        // std::cout << full_count << std::endl;
      }

      // CanDE infer implementation using associative memory,
      // this is fast only for small datasets
      void cande_infer_associative(const QueryType &q, int qid,
                                   int_fast64_t k, int_fast64_t num_probes,
                                   int_fast64_t max_num_candidates,
                                   std::vector<LSHTableKeyType> *result,
                                   std::vector<CoordinateType> *kde_result,
                                   CanDETask task)
      {
        if (result == nullptr || kde_result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }
        auto start_time = std::chrono::high_resolution_clock::now();

        FalconnRange &range = falconn_config.bins_vector[0];
        int num_bins = range.num_bins();
        int num_values;
        switch (task)
        {
        case CanDETask::kKDE:
          num_values = falconn_config.gamma.size();
          break;
        case CanDETask::kQDDE:
          num_values = num_bins;
          break;
        case CanDETask::kRecall:
          num_values = 1;
          break;
        default:
          throw NearestNeighborQueryError("Unknown CanDE task.");
        }
        int num_sums = (task == CanDETask::kKDE) ? falconn_config.gamma.size() : 1;
        std::vector<int> candidate_num_per_table;
        int num_tables = falconn_config.hash_table_params[0].l;

        std::string prob_filename = falconn_config.prob_filename + "_" + std::to_string(qid) + ".fvecs";

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();
        kde_result->assign(num_values, 0);

        // Step 1: get duplicate candidates
        table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                                     &candidates_, &candidate_num_per_table);
        k_ = k;
        initialize_heap();

        std::vector<double> recalls(num_bins, 0.0);
        std::vector<LSHTableKeyType> dedup_candidates;
        dedup_candidates.reserve(candidates_.size());
        std::vector<float> distances(falconn_config.num_points);
        // stores the table in which this candidate first appears
        std::vector<uint8_t> cand_tables(falconn_config.num_points, 255);
        std::vector<int> uniq_cands_per_bin(num_bins, 0);
        // candidates by table
        std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));
        // # candidates unique to a given table
        auto utcands_per_bin = tcands_per_bin;
        std::vector<std::vector<float>> kde_sums(num_sums, std::vector<float>(num_bins, 0.0));
        std::vector<float> gammas(num_values);
        if (task == CanDETask::kKDE)
        {
          std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
                         [](float x)
                         { return x * x * 2.0; });
        }

        auto distance_start_time = std::chrono::high_resolution_clock::now();
        // Step 2z: deduplicate
        int table = 0, count = 0;
        for (auto cand : candidates_)
        {
          while (count >= candidate_num_per_table[table])
            ++table;
          if (cand_tables[cand] == 255)
          { // new item {
            cand_tables[cand] = table;
            dedup_candidates.push_back(cand);
          }
          ++count;
        }

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(dedup_candidates);

        // Step 2a: compute all distances
        std::string distance_filename = std::vformat(falconn_config.distance_filename,
                                                     std::make_format_args("dedup", qid));
        if (falconn_config.load_distance && std::filesystem::exists(distance_filename))
        {
          std::vector<float> distance_buffer = read_data<std::vector<float>>(distance_filename)[0];
          // otherwise the distance file may be corrupted
          assert(distance_buffer.size() == dedup_candidates.size());
          auto dist_iter = distance_buffer.begin();
          while (iter.is_valid())
          {
            int key = iter.get_key();
            float dist = *dist_iter;
            distances[key] = dist;
            insert_heap(dist, key);
            ++iter;
            ++dist_iter;
          }
        }
        else
        {
          std::vector<float> distance_buffer;
          while (iter.is_valid())
          {
            DistanceType distance = dst_(q, iter.get_point());
            distances[iter.get_key()] = distance;
            if (falconn_config.load_distance)
            {
              distance_buffer.push_back(distance);
            }
            // here candidates are deduplicated
            insert_heap(distance, iter.get_key());
            ++iter;
          }
          if (falconn_config.load_distance)
          {
            std::ofstream fout(distance_filename);
            if (!fout)
            {
              std::filesystem::create_directories(
                  std::filesystem::path(distance_filename).parent_path());
              fout.close();
              fout.open(distance_filename);
            }
            write_data(fout, distance_buffer);
          }
        }

        // Step 2b: deduplicate using hash tables, and update inference counters
        table = 0;
        count = 0;
        for (auto key : candidates_)
        {
          while (count >= candidate_num_per_table[table])
            ++table;
          int bin = range.bin_translate(distances[key]);
          if (cand_tables[key] == table)
          {
            // first occurrence of candidate
            ++tcands_per_bin[bin][table];
            ++utcands_per_bin[bin][table];
            ++uniq_cands_per_bin[bin];
            if (task == CanDETask::kKDE)
            {
              for (int gammaid = 0; gammaid < num_values; ++gammaid)
              {
                kde_sums[gammaid][bin] += ONIAK::kdev2(distances[key], gammas[gammaid]);
              }
            }
            else
            {
              kde_sums[0][bin] += 1.0;
            }
          }
          else
          { // duplicate
            ++tcands_per_bin[bin][table];
            if (cand_tables[key] != 254)
            {
              --utcands_per_bin[bin][cand_tables[key]];
            }
            cand_tables[key] = 254;
          }
          ++count;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        stats_.average_distance_time += elapsed_distance.count();

        // step 3: return top Knn results
        dump_knn(res);

        // step 4: infer recalls of each bin
        std::vector<std::vector<double>> rhos_bin_table;
        for (int bin = 0; bin < num_bins; ++bin)
        {
          std::vector<double> rhos(num_tables, 0.0);
          // if (falconn_config.hypergeometric_bayesian) {
          //   double num_balls = uniq_cands_per_bin[bin] / mp_recalls_[range.mid_point(bin)];
          //   for (int round = 0; round < 10; ++round) {
          //     rhos.clear();
          //     for (int table = 0; table < num_tables; ++table) {
          //       double infer_result = bayesian_inference2(tcands_per_bin[bin][table],
          //                                        utcands_per_bin[bin][table],
          //                                        num_balls,
          //                                        uniq_cands_per_bin[bin]);
          //       if (infer_result > 0) rhos.push_back(infer_result);
          //     }
          //     num_balls = uniq_cands_per_bin[bin] / ONIAK::final_prob(rhos);
          //     if (qid == 0 && bin == 1) {
          //       std::cout << "round " << round << " " << num_balls << std::endl;
          //       using namespace ONIAK;
          //       std::cout << rhos << std::endl;
          //     }
          //   }
          //   recalls[bin] = uniq_cands_per_bin[bin] / num_balls;
          // } else {
          for (int table = 0; table < num_tables; ++table)
          {
            rhos[table] = bayesian_inference(tcands_per_bin[bin][table],
                                             utcands_per_bin[bin][table],
                                             uniq_cands_per_bin[bin],
                                             range.mid_point(bin));
          }
          rhos_bin_table.push_back(rhos);
          recalls[bin] = ONIAK::final_prob(rhos);
          // }
        }

        std::ofstream val_out(prob_filename);
        for (auto &vec : rhos_bin_table)
        {
          write_data(val_out, vec);
        }

        // step 5: infer KDE or QDDE
        if (task == CanDETask::kKDE)
        {
          for (int gammaid = 0; gammaid < num_sums; ++gammaid)
          {
            for (int bin = 0; bin < num_bins; ++bin)
            {
              if (recalls[bin] > 1e-4)
              { // avoid division by zero
                kde_result->operator[](gammaid) += kde_sums[gammaid][bin] / recalls[bin];
              }
            }
          }
        }
        else
        {
          if (task == CanDETask::kQDDE)
          {
            for (int bin = 0; bin < num_bins; ++bin)
            {
              if (recalls[bin] > 1e-4)
              { // avoid division by zero
                kde_result->operator[](bin) += kde_sums[0][bin] / recalls[bin];
              }
            }
          }
          else
          { // task is recall
            double xseen = 0., xfound = 0.;
            for (int bin = 0; bin < num_bins; ++bin)
            {
              if (recalls[bin] > 1e-4)
              { // avoid division by zero
                double new_seen = kde_sums[0][bin] / recalls[bin];
                if (xseen + new_seen < k_)
                {
                  xseen += new_seen;
                  xfound += uniq_cands_per_bin[bin];
                }
                else
                {
                  // "unbiased" estimator
                  xfound += (k_ - xseen) * recalls[bin];
                  break;
                }
              }
            }
            kde_result->operator[](0) = xfound / k_;
          }
        }

        auto sketches_end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_sketches =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                sketches_end_time - end_time);
        stats_.average_sketches_time += elapsed_sketches.count();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(sketches_end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
        // std::cout << full_count << std::endl;
      }

      void knn_and_qdde_infer(const QueryType &q, int qid,
                              int_fast64_t k, int_fast64_t num_probes,
                              int_fast64_t max_num_candidates,
                              std::vector<LSHTableKeyType> *result,
                              std::vector<CoordinateType> *result_infer,
                              CanDETask task)
      {
        if (result == nullptr || result_infer == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }
        if (task == CanDETask::kKDE)
        {
          throw NearestNeighborQueryError("KDE not supported for this method.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        FalconnRange &range = falconn_config.bins_vector[0];
        // std::string middle_filename = falconn_config.middle_result_filename + "_" + std::to_string(qid) + ".fvecs";
        int num_tables = falconn_config.hash_table_params[0].l;
        int num_bins = range.num_bins();
        table_bin_ht_.clear();
        std::vector<int> candidate_num_per_table;

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        // Step 1: get duplicate candidates
        table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                                     &candidates_, &candidate_num_per_table);
        k_ = k;
        initialize_heap();
        int result_infer_size = (task == CanDETask::kQDDE) ? num_bins : 1;
        result_infer->assign(result_infer_size, 0.0);

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);
        std::vector<float> distances;
        distances.reserve(candidates_.size());
        std::vector<int> uniq_cands_per_bin(num_bins, 0);
        // candidates by table
        std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));
        // # candidates unique to a given table
        auto utcands_per_bin = tcands_per_bin;

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        // Step 2a: compute all distances
        std::string distance_filename = std::vformat(falconn_config.distance_filename,
                                                     std::make_format_args("dup", qid));
        if (falconn_config.load_distance && std::filesystem::exists(distance_filename))
        {
          distances = read_data<std::vector<float>>(distance_filename)[0];
        }
        else
        {
          distances.reserve(candidates_.size());
          while (iter.is_valid())
          {
            DistanceType distance = dst_(q, iter.get_point());
            distances.push_back(distance);
            ++iter;
          }
        }
        if (falconn_config.load_distance && !std::filesystem::exists(distance_filename))
        {
          std::ofstream fout(distance_filename);
          if (!fout)
          {
            std::filesystem::create_directories(
                std::filesystem::path(distance_filename).parent_path());
            fout.close();
            fout.open(distance_filename);
          }
          write_data(fout, distances);
        }

        int table = 0, count = 0;
        // Step 2b: deduplicate using hash tables, and update inference counters
        for (auto key : candidates_)
        {
          ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
          float distance = distances[count];
          int bin = range.bin_translate(distance);
          while (count >= candidate_num_per_table[table])
            ++table;
          auto &table_find = table_bin_ht_.find_or_insert(key, status);
          // ONIAK::TableBinPair table_bin = {2,2};
          if (status == ONIAK::HashInsertionStatus::kNewlyInserted)
          {
            // first occurrence of candidate
            // Bin
            // table_bin.bin = bin;
            table_find = table;
            ++tcands_per_bin[bin][table];
            ++utcands_per_bin[bin][table];
            ++uniq_cands_per_bin[bin];
            insert_heap(distance, key);
          }
          else if (status == ONIAK::HashInsertionStatus::kAlreadyExists)
          {
            ++tcands_per_bin[bin][table];
            if (table_find != 255)
            {
              --utcands_per_bin[bin][table_find];
              table_find = 255;
            }
          }
          else
          { // if hash table is full, only check for knn
            if (num_inserted_ < 0)
            {
              insert_heap(distance, key);
            } // otherwise the result may contain duplicates.
          }
          ++count;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        stats_.average_distance_time += elapsed_distance.count();

        // step 3: return top Knn results
        dump_knn(res);
        std::vector<std::vector<double>> rhos_bin_table;

        // step 4: infer histogram of each bin
        double xseen = 0., xfound = 0.;
        for (int bin = 0; bin < num_bins; ++bin)
        {
          if (uniq_cands_per_bin[bin] == 0)
            continue;

          std::vector<double> rhos(num_tables, 0.0);
          for (int table = 0; table < num_tables; ++table)
          {
            rhos[table] = bayesian_inference(tcands_per_bin[bin][table],
                                             utcands_per_bin[bin][table],
                                             uniq_cands_per_bin[bin],
                                             range.mid_point(bin));
          }
          rhos_bin_table.push_back(rhos);
          double rho = ONIAK::final_prob(rhos);
          if (task == CanDETask::kQDDE)
          {
            result_infer->operator[](bin) = uniq_cands_per_bin[bin] / rho;
          }
          else
          { // recall
            double new_seen = uniq_cands_per_bin[bin] / rho;
            if (xseen + new_seen < k_)
            {
              xseen += new_seen;
              xfound += uniq_cands_per_bin[bin];
            }
            else
            {
              // "unbiased" estimator
              xfound += (k_ - xseen) * rho;
              break;
            }
          }
        }
        if (task == CanDETask::kRecall)
        {
          result_infer->operator[](0) = xfound / k_;
        }

        auto sketches_end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_sketches =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                sketches_end_time - end_time);
        stats_.average_sketches_time += elapsed_sketches.count();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(sketches_end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      // CanDE inference adjusted by CP
      void knn_and_cande_cp_adjusted(const QueryType &q, int qid,
                                     int_fast64_t k, int_fast64_t num_probes,
                                     int_fast64_t max_num_candidates,
                                     std::vector<LSHTableKeyType> *result,
                                     std::vector<CoordinateType> *result_infer,
                                     CanDETask task)
      {
        if (result == nullptr || result_infer == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        FalconnRange &range = falconn_config.bins_vector[0];
        int num_tables = falconn_config.hash_table_params[0].l;
        int num_bins = range.num_bins();
        table_bin_ht_.clear();
        std::vector<int> candidate_num_per_table;
        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        // Step 1: get duplicate candidates
        table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                                     &candidates_, &candidate_num_per_table);
        k_ = k;
        initialize_heap();
        int num_values;
        switch (task)
        {
        case CanDETask::kKDE:
          num_values = falconn_config.gamma.size();
          break;
        case CanDETask::kQDDE:
          num_values = num_bins;
          break;
        case CanDETask::kRecall:
          num_values = 1;
          break;
        default:
          throw NearestNeighborQueryError("Unknown CanDE task.");
        }
        result_infer->assign(num_values, 0.0);
        std::vector<float> gammas(num_values);
        if (task == CanDETask::kKDE)
        {
          std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
                         [](float x)
                         { return x * x * 2.0; });
        }

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);
        std::vector<float> distances;
        distances.reserve(candidates_.size());

        // tcands_per_bin, uniq_cands_per_bin, utcands_per_bin are all weight-adjusted
        std::vector<float> uniq_cands_per_bin(num_bins, 0);
        // candidates by table
        std::vector<std::vector<float>> tcands_per_bin(num_bins, std::vector<float>(num_tables, 0));
        // # candidates unique to a given table
        auto utcands_per_bin = tcands_per_bin;

        // For KDE, this stores the sum of kernel densities per bin
        std::vector<std::vector<float>> kde_sum(num_bins, std::vector<float>(falconn_config.gamma.size(), 0));
        // For QDDE and recall, it stores the number of unique candidates per bin (not adjusted)
        std::vector<int> bin_counters(num_bins, 0);

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        // Step 2a: compute all distances
        // TODO: Use Bloom filter to avoid duplicate distance computation
        std::string distance_filename = std::vformat(falconn_config.distance_filename,
                                                     std::make_format_args("dup", qid));
        if (falconn_config.load_distance && std::filesystem::exists(distance_filename))
        {
          distances = read_data<std::vector<float>>(distance_filename)[0];
        }
        else
        {
          distances.reserve(candidates_.size());
          while (iter.is_valid())
          {
            DistanceType distance = dst_(q, iter.get_point());
            distances.push_back(distance);
            ++iter;
          }
        }
        if (falconn_config.load_distance && !std::filesystem::exists(distance_filename))
        {
          std::ofstream fout(distance_filename);
          if (!fout)
          {
            std::filesystem::create_directories(
                std::filesystem::path(distance_filename).parent_path());
            fout.close();
            fout.open(distance_filename);
          }
          write_data(fout, distances);
        }

        int table = 0, count = 0;
        // Step 2b: deduplicate using hash tables, and update inference counters
        for (auto key : candidates_)
        {
          ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
          float distance = distances[count];
          int bin = range.bin_translate(distance);
          while (count >= candidate_num_per_table[table])
            ++table;
          auto &table_find = table_bin_ht_.find_or_insert(key, status);
          // ONIAK::TableBinPair table_bin = {2,2};
          if (status == ONIAK::HashInsertionStatus::kNewlyInserted)
          {
            // first occurrence of candidate
            // Bin
            // table_bin.bin = bin;
            table_find = table;
            insert_heap(distance, key);

            float adjusted_weight = 1.0 / mp_recalls_[distance];
            tcands_per_bin[bin][table] += adjusted_weight;
            utcands_per_bin[bin][table] += adjusted_weight;
            uniq_cands_per_bin[bin] += adjusted_weight;

            if (task == CanDETask::kKDE)
            {
              for (size_t gammaid = 0; gammaid < gammas.size(); ++gammaid)
              {
                kde_sum[bin][gammaid] += ONIAK::kdev2(distance, gammas[gammaid]);
              }
            }
            bin_counters[bin] += 1;
          }
          else if (status == ONIAK::HashInsertionStatus::kAlreadyExists)
          {
            float adjusted_weight = 1.0 / mp_recalls_[distance];
            tcands_per_bin[bin][table] += adjusted_weight;
            if (table_find != 255)
            {
              utcands_per_bin[bin][table_find] -= adjusted_weight;
              table_find = 255;
            }
          } // if hash table is full, drop this candidate
          ++count;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        stats_.average_distance_time += elapsed_distance.count();

        // step 3: return top Knn results
        dump_knn(res);
        std::vector<std::vector<double>> rhos_bin_table;

        // step 4: infer histogram of each bin
        double xseen = 0., xfound = 0.;
        for (int bin = 0; bin < num_bins; ++bin)
        {
          if (bin_counters[bin] == 0)
            continue;
          std::vector<double> rhos(num_tables, 0.0);
          for (int table = 0; table < num_tables; ++table)
          {
            rhos[table] = bayesian_inference(tcands_per_bin[bin][table],
                                             utcands_per_bin[bin][table],
                                             uniq_cands_per_bin[bin],
                                             range.mid_point(bin));
          }
          rhos_bin_table.push_back(rhos);
          double rho = ONIAK::final_prob(rhos);
          if (task == CanDETask::kQDDE)
          {
            result_infer->operator[](bin) = bin_counters[bin] / rho;
          }
          else if (task == CanDETask::kKDE)
          {
            for (size_t gammaid = 0; gammaid < falconn_config.gamma.size(); ++gammaid)
            {
              result_infer->operator[](gammaid) += kde_sum[bin][gammaid] / rho;
            }
          }
          else
          { // recall
            double new_seen = bin_counters[bin] / rho;
            if (xseen + new_seen < k_)
            {
              xseen += new_seen;
              xfound += bin_counters[bin];
            }
            else
            {
              // "unbiased" estimator
              xfound += (k_ - xseen) * rho;
              break;
            }
          }
        }
        if (task == CanDETask::kRecall)
        {
          result_infer->operator[](0) = xfound / k_;
        }

        auto sketches_end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_sketches =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                sketches_end_time - end_time);
        stats_.average_sketches_time += elapsed_sketches.count();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(sketches_end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      // unbiased CanDE inference by resampling candidates under the same CP
      // Resample is deprecated due to poor performance
      void knn_and_cande_resample(const QueryType &q, int qid,
                                  int_fast64_t k, int_fast64_t num_probes,
                                  int_fast64_t max_num_candidates,
                                  std::vector<LSHTableKeyType> *result,
                                  std::vector<CoordinateType> *result_infer,
                                  CanDETask task) {}

      // Performs inference from precomputed collision probability
      void cande_precomputed(const QueryType &q, int qid,
                             int_fast64_t k, int_fast64_t num_probes,
                             int_fast64_t max_num_candidates,
                             std::vector<LSHTableKeyType> *result,
                             std::vector<CoordinateType> *val_result,
                             CanDETask task)
      {
        if (result == nullptr || val_result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();
        FalconnRange &range = falconn_config.bins_vector[0];
        int num_bins = range.num_bins();
        int num_values;
        switch (task)
        {
        case CanDETask::kKDE:
          num_values = falconn_config.gamma.size();
          break;
        case CanDETask::kQDDE:
          num_values = num_bins;
          break;
        case CanDETask::kRecall:
          num_values = 1;
          break;
        default:
          throw NearestNeighborQueryError("Unknown CanDE task.");
        }

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);

        k_ = k;
        initialize_heap();
        val_result->assign(num_values, 0);

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);
        std::vector<float> distances;
        std::string distance_filename = std::vformat(falconn_config.distance_filename,
                                                     std::make_format_args("dedup", qid));
        if (falconn_config.load_distance && std::filesystem::exists(distance_filename))
        {
          auto distances_load = read_data<std::vector<float>>(distance_filename);
          assert(distances_load.size() == 1);
          distances = std::move(distances_load[0]);
          assert(distances.size() == candidates_.size());
          auto dist_iter = distances.begin();
          while (iter.is_valid())
          {
            insert_heap(*dist_iter, iter.get_key());
            ++iter;
            ++dist_iter;
          }
        }
        else
        {
          distances.reserve(candidates_.size());
          while (iter.is_valid())
          {
            DistanceType distance = dst_(q, iter.get_point());
            distances.push_back(distance);
            // here candidates are deduplicated
            insert_heap(distance, iter.get_key());
            ++iter;
          }
        }
        if (falconn_config.load_distance && !std::filesystem::exists(distance_filename))
        {
          std::ofstream fout(distance_filename);
          if (!fout)
          {
            std::filesystem::create_directories(
                std::filesystem::path(distance_filename).parent_path());
            fout.close();
            fout.open(distance_filename);
          }
          write_data(fout, distances);
        }

        std::vector<float> gammas(num_values);
        if (task == CanDETask::kKDE)
        {
          std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
                         [](float x)
                         { return x * x * 2.0; });
        }

        for (float cur_distance : distances)
        {
          switch (task)
          {
          case CanDETask::kKDE:
            for (size_t gammaid = 0; gammaid < val_result->size(); ++gammaid)
            {
              val_result->operator[](gammaid) +=
                  ONIAK::kdev2(cur_distance, gammas[gammaid]) / mp_recalls_[cur_distance];
            }
            break;
          case CanDETask::kQDDE:
          {
            int bin = range.bin_translate(cur_distance);
            val_result->operator[](bin) += 1.0 / mp_recalls_[cur_distance];
          }
          break;
          default:
            break;
          }
        }

        dump_knn(res);
        if (task == CanDETask::kRecall)
        {
          double xseen = 0., xfound = 0.;
          for (int kk = 0; kk < k_; ++kk)
          {
            int ki = k_ - kk - 1;
            double dist = -heap_.get_data()[ki].key; // minus distance are used as keys
            double new_seen = 1.0 / mp_recalls_[dist];
            if (xseen + new_seen < k_)
            {
              xseen += new_seen;
              xfound++;
            }
            else
            {
              // "unbiased" estimator
              xfound += (k_ - xseen) * mp_recalls_[dist];
              break;
            }
          }
          val_result->operator[](0) = xfound / k_;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      // Function to output candidates per table
      std::vector<int> knn_candidate(const QueryType &q,
                                     int_fast64_t k, int_fast64_t num_probes,
                                     int_fast64_t max_num_candidates)
      {
        std::vector<int> candidate_num_per_table;

        // Step 1: get duplicate candidates
        table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                                     &candidates_, &candidate_num_per_table);

        return candidate_num_per_table;
      }

      void find_near_neighbors(const QueryType &q,
                               const QueryType &q_comp,
                               DistanceType threshold, int_fast64_t num_probes,
                               int_fast64_t max_num_candidates,
                               std::vector<LSHTableKeyType> *result)
      {
        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);
        auto distance_start_time = std::chrono::high_resolution_clock::now();

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);
        while (iter.is_valid())
        {
          DistanceType cur_distance = dst_(q_comp, iter.get_point());
          if (cur_distance < threshold)
          {
            res.push_back(iter.get_key());
          }
          ++iter;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      void get_candidates_with_duplicates(const QueryType &q,
                                          int_fast64_t num_probes,
                                          int_fast64_t max_num_candidates,
                                          std::vector<FalconnCandidateType> *result)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_duplicate_candidates_and_tables(q, num_probes,
                                                          max_num_candidates, result);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      void get_unique_candidates(const QueryType &q,
                                 int_fast64_t num_probes,
                                 int_fast64_t max_num_candidates,
                                 std::vector<FalconnCandidateType> *result)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_unique_candidates_and_tables(q, num_probes, max_num_candidates,
                                                       result);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      void reset_query_statistics()
      {
        table_query_->reset_query_statistics();
        stats_.reset();
      }

      QueryStatistics get_query_statistics()
      {
        QueryStatistics res = table_query_->get_query_statistics();
        res.average_total_query_time = stats_.average_total_query_time;
        res.average_distance_time = stats_.average_distance_time;
        res.average_sketches_time = stats_.average_sketches_time;

        if (res.num_queries > 0)
        {
          res.average_total_query_time /= res.num_queries;
          res.average_distance_time /= res.num_queries;
          res.average_sketches_time /= res.num_queries;
        }
        return res;
      }

      FalconnMultiprobeType get_transformed_vector(const QueryType &q)
      {
        return table_query_->get_transformed_vector(q);
      }

      FalconnProbingListType get_probing_sequence(const QueryType &q)
      {
        return table_query_->get_probing_sequence(q);
      }

    private:
      LSHTableQuery *table_query_;
      const DataStorage &data_storage_;
      std::vector<LSHTableKeyType> candidates_;
      DistanceFunction dst_;
      SimpleHeap<DistanceType, LSHTableKeyType> heap_;
      // Precompute multiprobed recalls for L, 1, and L-1 LSH tables
      ONIAK::RealArray mp_recalls_, mp_recalls1_, mp_recalls_lm1_;

      ONIAK::ONIAKHT<int, uint8_t, uint16_t, 64> table_bin_ht_;
      ONIAK::ONIAKHT<int, dist_table_pair, uint32_t, 96> knn_tables_;

      // used in resampling of candidates
      ONIAK::WYHash hash_func_;
      QueryStatistics stats_;
      // number of nearest neighbors to return
      int k_, num_inserted_;
      double bayesian_prior_;
      std::vector<float> cps_bin_per_table_;

      void initialize_heap()
      {
        num_inserted_ = 0;
        heap_.reset();
        heap_.resize(k_);
      }

      void insert_heap(DistanceType distance, LSHTableKeyType key)
      {
        if (num_inserted_ >= 0)
        {
          heap_.insert_unsorted(-distance, key);
          ++num_inserted_;
          if (num_inserted_ == k_)
          {
            heap_.heapify();
            num_inserted_ = -1;
          }
        }
        else if (distance < -heap_.min_key())
        {
          heap_.replace_top(-distance, key);
        }
      }

      void dump_knn(std::vector<LSHTableKeyType> &res)
      {
        int heap_size = (num_inserted_ < 0) ? k_ : num_inserted_;
        res.resize(heap_size);
        std::sort(heap_.get_data().begin(),
                  heap_.get_data().begin() + heap_size);
        for (int_fast64_t ii = 0; ii < heap_size; ++ii)
        {
          res[ii] = heap_.get_data()[heap_size - ii - 1].data;
        }
      }

      double bayesian_inference(int tcands, int utcands, int uniq_cands, double bin_mid)
      {
        // bayesian part of the remaining nearest neighbors
        double p1 = (tcands - utcands + bayesian_prior_ * mp_recalls1_[bin_mid]) /
                    (uniq_cands - utcands + bayesian_prior_);
        return p1;
      }

      // deprecated due to bad performance
      // double bayesian_inference2(int tcands, int utcands, double e, int uniq_cands) {
      //   // empirical collision rate
      //   double n_red_balls = tcands;
      //   double n_revealed_red_balls = tcands - utcands;
      //   double n_revealed_balls = uniq_cands - utcands;
      //   // if (n_revealed_red_balls < 1) return -1;
      //   // return n_red_balls * n_revealed_balls / n_revealed_red_balls;
      //   // int n_revealed_black_balls = n_revealed_balls - n_revealed_red_balls;

      //   //The conjugate prior distribution for number of black balls is hypergeometric(A, B, C)
      //   // with A - B - C = d = n_red_balls - nrevealed_balls
      //   // assume B / C = k, a parameter
      //   // double k = static_cast<double>(n_red_balls) / n_revealed_balls;
      //   // k = (k < 1.0) ? 1.0 / k : k;
      //   // k = (k > 30.0)? 30.0 : k;
      //   double k = falconn_config.bayesian_prior;
      //   double d = n_red_balls - n_revealed_balls;
      //   // expectation of prior distribution is BC/A = n_reveal_balls / CP - n_red_balls;
      //   e -= n_red_balls;

      //   // solve quadratic equation for C and A, B
      //   // in principle these numbers are integers, but we do not round here
      //   double delta = e * e * (k + 1.0) * (k + 1.0) + 4 * k * d * e;
      //   double C = (e * (k + 1.0) + sqrt(delta)) / (2 * k);
      //   double B = k * C;
      //   double A = B + C + d;

      //   // posterior distribution for the number of balls is hypergeometric(A', B', C') with
      //   double A_prime = A + n_revealed_red_balls;
      //   double B_prime = B + n_red_balls;
      //   double C_prime = C + n_red_balls;

      //   double n_balls = B_prime * C_prime / A_prime;
      //   // return inferred ratio of red balls
      //   return n_red_balls / n_balls;
      // }
    };

  } // namespace core
} // namespace falconn

#endif
