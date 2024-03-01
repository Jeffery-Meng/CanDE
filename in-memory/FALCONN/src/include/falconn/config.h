#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <fstream>
#include <algorithm>
#include <memory>
#include <random>
#include <cmath>
#include <limits>
#include "nlohmann/json.hpp"
#include "lsh_nn_table.h"
#include "core/math_helpers.h"

namespace falconn
{

  using json = nlohmann::json;

  LSHConstructionParameters read_config(const char *filename)
  {
    std::ifstream json_f(filename);
    json config;
    json_f >> config;

    LSHConstructionParameters para;

    para.dimension = config.at("dimension");
    if (config.contains("seed"))
    {
      para.seed = config["seed"];
      para.seeding_sequence = std::make_unique<std::seed_seq>(
          std::initializer_list<uint64_t>({para.seed}));
    }
    if (config.contains("number of rotations"))
    {
      para.num_rotations = config.at("number of rotations");
    }
    if (config.contains("hash table width"))
    {
      para.hash_table_width = config.at("hash table width");
    }
    if (config.contains("save index"))
    {
      para.save_index = config.at("save index");
    }
    if (config.contains("index filename"))
    {
      para.index_filename = static_cast<std::string>(config["index filename"]);
    }
    if (config.contains("index record path"))
    {
      para.index_record_path = static_cast<std::string>(config["index record path"]);
    }
    if (config.contains("eigenvalue filepath"))
    {
      para.eigen_filename = config["eigenvalue filepath"];
    }
    if (config.contains("middle result filename"))
    {
      para.middle_result_filename = config["middle result filename"];
    }

    if (config.contains("number of partitions"))
    {
      para.num_partitions = config["number of partitions"];
    }

    auto hash_params = config.at("hash table parameters");
    int_fast32_t k_min = std::numeric_limits<int>::max();
    for (const auto &par_param : hash_params)
    {
      CompositeHashTableParameters ch_para;
      ch_para.k = par_param.at("k");
      k_min = std::min(k_min, ch_para.k);
      ch_para.l = par_param.at("l");
      if (par_param.contains("bucket width"))
      {
        ch_para.bucket_width = par_param.at("bucket width");
      }
      if (par_param.contains("lower"))
      {
        ch_para.bucket_width = par_param.at("lower");
      }
      if (par_param.contains("upper"))
      {
        ch_para.bucket_width = par_param.at("upper");
      }
      para.hash_table_params.push_back(ch_para);

      int kbyl = ch_para.k * ch_para.l;
      if (kbyl > para.num_hash_funcs)
      {
        para.num_hash_funcs = kbyl;
      }
    }

    std::sort(para.hash_table_params.begin(), para.hash_table_params.end(),
              [](const CompositeHashTableParameters &a, const CompositeHashTableParameters &b)
              {
                return a.partition_upper < b.partition_upper;
              });

    // if (config.contains("second step")) {
    //     para.second_step = config["second step"];
    // }
    para.second_step = false; // Enforce single step ONIAK

    if (config.contains("fast rotation"))
    {
      para.fast_rotation = config["fast rotation"];
    }
    if (config.contains("number of query rows"))
    {
      para.dim_Arows = config["number of query rows"];
    }

    para.rotation_dim = core::find_next_power_of_two(para.dimension);
    para.dim_Acols = para.dimension - 1;
    para.num_setup_threads = 1;

    if (config.contains("allow overwrite"))
    {
      para.allow_overwrite = config["allow overwrite"];
    }
    if (config.contains("ground truth file"))
    {
      para.gnd_filename = config["ground truth file"];
    }
    if (config.contains("compute ground truth"))
    {
      para.compute_gound_truth = config["compute ground truth"];
    }
    if (config.contains("number of neighbors"))
    {
      if (config["number of neighbors"].is_array())
      {
        for (auto nn_val : config["number of neighbors"])
        {
          para.num_neighbors_list.push_back(nn_val);
        }
      }
      else
      {
        para.num_neighbors = config["number of neighbors"];
        para.num_neighbors_list.push_back(para.num_neighbors);
      }
    }
    if (config.contains("training size"))
    {
      para.num_points = config["training size"];
    }
    if (config.contains("testing size"))
    {
      para.num_queries = config["testing size"];
    }
    if (config.contains("data filename"))
    {
      para.data_filename = config["data filename"];
    }
    if (config.contains("prob filename"))
    {
      para.prob_filename = config["prob filename"];
    }
    if (config.contains("query filename"))
    {
      para.query_filename = config["query filename"];
    }
    if (config.contains("kernel filename"))
    {
      para.kernel_filename = config["kernel filename"];
    }
    if (config.contains("distance filename"))
    {
      para.distance_filename = config["distance filename"];
    }
    if (config.contains("result filename"))
    {
      para.result_filename = config["result filename"];
    }
    if (config.contains("summary path"))
    { // print out summary
      // recall + candidate #, used for parameter tuning
      para.summary_path = config["summary path"];
    }
    if (config.contains("use single kernel"))
    {
      para.single_kernel = config["use single kernel"];
    }
    if (config.contains("input transformed queries"))
    {
      para.transformed_queries = config["input transformed queries"];
    }
    if (config.contains("candidate filename"))
    {
      para.candidate_filename = config["candidate filename"];
    }
    if (config.contains("number of prefilter"))
    {
      para.num_prehash_filters = config["number of prefilter"];
    }
    if (config.contains("ratio of prefilter"))
    {
      para.prefilter_ratio = config["ratio of prefilter"];
    }
    if (config.contains("row id filename"))
    {
      para.rowid_filename = config["row id filename"];
    }
    if (config.contains("query mode"))
    {
      std::string query_m = config["query mode"];
      if (query_m == "knn nearest neighbors")
      {
        para.query_mode = FalconnQueryMode::KNNWithTime;
      }
      else if (query_m == "knn candidates")
      {
        para.query_mode = FalconnQueryMode::PrintCandidates;
      }
      else if (query_m == "knn recall")
      {
        para.query_mode = FalconnQueryMode::KNNRecall;
      }
      else if (query_m == "hashed queries")
      {
        para.query_mode = FalconnQueryMode::PrintHashedQueries;
      }
      else if (query_m == "probing sequence")
      {
        para.query_mode = FalconnQueryMode::PrintProbingSequence;
      }
      else if (query_m == "hashed data")
      {
        para.query_mode = FalconnQueryMode::PrintHashedData;
      }
      else if (query_m == "knn duplicate candidates")
      {
        para.query_mode = FalconnQueryMode::PrintDuplicateCandidates;
      }
      else if (query_m == "Print candidates number")
      {
        para.query_mode = FalconnQueryMode::CandidateNum;
      }
      else if (query_m == "measure time")
      {
        para.query_mode = FalconnQueryMode::PrintTimeMeasurements;
      }
      else if (query_m == "precomputed sequence")
      {
        para.query_mode = FalconnQueryMode::PrintPrecomputedMPSequence;
      }
      else if (query_m == "hash function")
      {
        para.query_mode = FalconnQueryMode::PrintHashFunction;
      }
      else if (query_m == "knn time")
      {
        para.query_mode = FalconnQueryMode::MeasureKNNTime;
      }
      else if (query_m == "knn and kde infer")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kKDE;
        para.cande_algos = {CanDEAlgoType::kHash};
      }
      else if (query_m == "knn and qdde infer")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kQDDE;
        para.cande_algos = {CanDEAlgoType::kHash};
      }
      else if (query_m == "knn and kde precomputed")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kKDE;
        para.cande_algos = {CanDEAlgoType::kPrecompute};
      }
      else if (query_m == "knn and qdde precomputed")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kQDDE;
        para.cande_algos = {CanDEAlgoType::kPrecompute};
      }
      else if (query_m == "knn and kde associative")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kKDE;
        para.cande_algos = {CanDEAlgoType::kAssociative};
      }
      else if (query_m == "knn and qdde associative")
      {
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
        para.cande_task = CanDETask::kQDDE;
        para.cande_algos = {CanDEAlgoType::kAssociative};
      }
      else if (query_m == "cande")
      {
        // User must specify cande task and algorithm
        para.query_mode = FalconnQueryMode::KNNWithCanDE;
      }
      else if (query_m == "qdde ground truth")
      {
        para.query_mode = FalconnQueryMode::QDDEGroundTruth;
      }
    }
    if (config.contains("cande task"))
    {
      std::string cande_task = config["cande task"];
      if (cande_task == "KDE" || cande_task == "kde")
      {
        para.cande_task = CanDETask::kKDE;
      }
      else if (cande_task == "QDDE" || cande_task == "qdde")
      {
        para.cande_task = CanDETask::kQDDE;
      }
      else if (cande_task == "recall")
      {
        para.cande_task = CanDETask::kRecall;
      }
    }
    if (config.contains("cande algorithm"))
    {
      for (auto cande_algo : config["cande algorithm"])
      {
        if (cande_algo == "precompute")
        {
          para.cande_algos.push_back(CanDEAlgoType::kPrecompute);
        }
        else if (cande_algo == "associative")
        {
          para.cande_algos.push_back(CanDEAlgoType::kAssociative);
        }
        else if (cande_algo == "hash")
        {
          para.cande_algos.push_back(CanDEAlgoType::kHash);
        }
        else if (cande_algo == "weighted recall")
        {
          para.cande_algos.push_back(CanDEAlgoType::kWeightedRecall);
        }
        else if (cande_algo == "cp adjusted")
        {
          para.cande_algos.push_back(CanDEAlgoType::kCPAdjusted);
        }
      }
    }
    if (config.contains("probes per table"))
    {
      para.probes_per_table = config["probes per table"];
      para.probes_per_table = std::min(para.probes_per_table,
                                       static_cast<int>(std::pow(3, k_min)));
    }
    if (config.contains("printing mode"))
    {
      auto json_mode = config["printing mode"];
      if (json_mode.contains("id"))
      {
        para.printing_mode.print_id = json_mode["id"];
      }
      if (json_mode.contains("table"))
      {
        para.printing_mode.print_table = json_mode["table"];
      }
      if (json_mode.contains("bucket"))
      {
        para.printing_mode.print_bucket_order = json_mode["bucket"];
      }
      if (json_mode.contains("distance"))
      {
        para.printing_mode.print_distance = json_mode["distance"];
      }
    }
    if (config.contains("gamma"))
    {
      for (auto val : config["gamma"])
      {
        para.gamma.push_back(val);
      }
    }
    if (config.contains("query hash filename"))
    {
      para.query_hash_filename = config["query hash filename"];
    }
    if (config.contains("probing sequence file"))
    {
      para.probing_sequence_file = config["probing sequence file"];
    }
    if (config.contains("recall p filename"))
    {
      para.recall_p_filename = config["recall p filename"];
    }
    if (config.contains("mp prob filename"))
    {
      para.mp_prob_filename = config["mp prob filename"];
    }
    if (config.contains("number of experiments"))
    {
      para.num_experiments = config["number of experiments"];
    }
    if (config.contains("query variant"))
    {
      std::string query_v = config["query variant"];
      if (query_v == "top k")
      {
        para.query_variant = NNQueryVariant::TopK;
      }
      else if (query_v == "radius r")
      {
        para.query_variant = NNQueryVariant::RadiusR;
      }
    }
    if (config.contains("nn radius"))
    {
      for (auto val : config["nn radius"])
      {
        para.radius_R.push_back(val);
      }
    }
    if (config.contains("max attempts"))
    {
      para.max_attempts = config["max attempts"];
    }
    if (config.contains("average radius"))
    {
      para.avg_radius = config["average radius"];
    }
    if (config.contains("bayesian prior"))
    {
      para.bayesian_prior = config["bayesian prior"];
    }
    if (config.contains("hypergeometric bayesian"))
    {
      para.hypergeometric_bayesian = config["hypergeometric bayesian"];
    }
    if (config.contains("load distance"))
    {
      para.load_distance = config["load distance"];
    }
    if (config.contains("histogram bins"))
    {
      para.histogram_bins = {config["histogram bins"][0], config["histogram bins"][1],
                             config["histogram bins"][2]};
      int hist_size = config["histogram bins"].size();
      if (hist_size > 3)
      {
        assert(hist_size % 4 == 0);
        for (auto iter = config["histogram bins"].begin(); iter != config["histogram bins"].end();
             iter += 4)
        {
          para.bins_vector.emplace_back(*iter, *(iter + 1), *(iter + 2), *(iter + 3));
        }
      }
    }
    if (config.contains("hash function filename"))
    {
      para.hash_func_filename = config["hash function filename"];
    }
    if (config.contains("histogram filename"))
    {
      para.histogram_filename = config["histogram filename"];
    }
    if (config.contains("tau"))
    {
      para.tau = config["tau"];
    }
    if (config.contains("target error"))
    {
      para.target_error = config["target error"];
    }
    if (config.contains("error range"))
    {
      para.error_range = config["error range"];
    }
    if (config.contains("dataset name"))
    {
      para.dataset = config["dataset name"];
    }
    if (config.contains("knn filename"))
    {
      para.knn_filename = config["knn filename"];
    }
    if (config.contains("kde filename"))
    {
      para.kde_filename = config["kde filename"];
    }
    if (config.contains("accuracy filename"))
    {
      para.accuracy_filename = config["accuracy filename"];
    }
    if (config.contains("accuracy binary filename"))
    {
      para.accuracy_binary_filename = config["accuracy binary filename"];
    }
    if (config.contains("cande table width"))
    {
      para.cande_table_size = 1u << static_cast<unsigned>(config["cande table width"]);
    }
    return para;
  }

}

#endif