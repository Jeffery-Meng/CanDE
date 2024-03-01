#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "nlohmann/json.hpp"

namespace ONIAK {

struct KroneckerExperimentConfig {
    int num_experiments, num_hashes, dim, probe_buckets, k;
    std::vector<DType> cosines;
    std::string output_path = "result.txt";
};

// Reads config from input path, also returns a hash value
KroneckerExperimentConfig read_config(const char * filename, size_t* hash_value) {
    std::ifstream json_f(filename);
    nlohmann::json config;
    json_f >> config;
    if (hash_value != nullptr) {
        *hash_value = std::hash<nlohmann::json>{}(config);
    }
    KroneckerExperimentConfig experiment;

    experiment.num_experiments = config.at("number of experiments");
    experiment.num_hashes = config.at("number of hash functions");
    auto& cosine_list = config.at("cosine values");
    experiment.cosines.assign(cosine_list.begin(), cosine_list.end());
    experiment.dim = config.at("dimension");

    if (config.contains("output path")) {
        experiment.output_path = config["output path"];
    }
    if (config.contains("probe buckets")) {
        experiment.probe_buckets = config["probe buckets"];
    }
    if (config.contains("hashes per table")) {
        experiment.k = config["hashes per table"];
    }
    return experiment;
}

}  // namespace ONIAK

#endif  // __CONFIG_HPP__