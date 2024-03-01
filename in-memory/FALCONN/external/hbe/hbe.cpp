/*
 *  Sketching experiments:
 *      Compare the relative error of Uniform, HBS, Herding and SKA under varying sketch sizes.
 *      For each sketch, we output the average relative error and stanfard error of the mean.
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 */

#include <chrono>
#include <memory>
#include "falconn/config.h"
#include "falconn/fileio.h"
#include "alg/RS.h"
#include "alg/HBS.h"
#include "alg/Herding.h"
#include "alg/KCenter.h"
#include "utils/DataIngest.h"
#include "parseConfig.h"

using namespace falconn;

constexpr double sqrt2 = 1.414213562373095;
constexpr int kNumIters = 5;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    // char* scope = argv[2];
    LSHConstructionParameters conf = read_config(argv[2]);
    const char* scope = conf.dataset.c_str();
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, /*read_exact=*/ false);
    data.estimateHashParams();

    //Will be used to obtain a seed for the random number engine
    //std::random_device rd;
    
    std::mt19937_64 rng(conf.seed);
    int num_samples = conf.num_points * conf.prefilter_ratio;
    vector<int> nsamples(1, num_samples);
    Eigen::MatrixXd kde_gt = ONIAK::read_file<double>(conf.gnd_filename);
    MatrixType distances = ONIAK::read_file<float>(conf.distance_filename);

    std::vector<double> re_mres(conf.gamma.size(), 0.0), hbs_mres(conf.gamma.size(), 0.0);
    auto selected_queries = read_data<std::vector<int>>(conf.rowid_filename)[0];
    HashSet<int> selected_qset(selected_queries.begin(), selected_queries.end());
    std::ofstream fout(conf.result_filename);

    for (size_t idx = 0; idx < nsamples.size(); idx ++) {
        int m = nsamples[idx];
        std::cout << "----------------------------" << std::endl;
        std::cout << "sketch size=" << m << std::endl;
        for (size_t iter = 0; iter < kNumIters; iter ++) {
            RS rs(data.X_ptr, data.kernel, m);
            HBS hbs_simple = HBS(data.X_ptr, m, data.w, data.k, 5, rng);
            auto& hbs_samples = hbs_simple.final_samples;
            DenseMatrix<double> hbe_estimators(conf.gamma.size(), conf.num_queries);
            
            // size_t dot_pos = conf.result_filename.find(".");
            // auto substr = conf.result_filename.substr(0, dot_pos);
            // ::ofstream dump_file(substr + "_iter_" + std::to_string(iter) + ".dvecs");
            
            for (size_t gammaid = 0; gammaid < conf.gamma.size(); ++gammaid) {
                auto gamma = conf.gamma[gammaid];

                for (int qid  = 0; qid < conf.num_queries; ++qid) {
                    // Precomputed Ground truth kernel density
                    double exact_val = kde_gt(gammaid, qid);

                    double hbs_est = 0;
                    for (size_t j = 0; j < hbs_samples.size(); j ++) {
                        int cid = hbs_samples[j].first;
                        hbs_est += hbs_samples[j].second * data.kernel->density(distances(qid, cid) / gamma / sqrt2);
                    }
                    hbs_est *= conf.num_points / hbs_samples.size();
                    hbe_estimators(gammaid, qid) = hbs_est;

                    if (selected_qset.find(qid) == selected_qset.end())  continue;
                    hbs_mres[gammaid] += std::fabs(hbs_est - exact_val) / exact_val;

                    // Random Sampling
                    //double rs_est = rs.query_precomputed(qid, data.tau, m, gamma, distances);
                    //rs_est *= conf.num_points;
                    //re_mres[gammaid] += std::fabs(rs_est - exact_val) / exact_val;
                }

                
            }

            std::cout << "# queries: " << selected_queries.size() << std::endl;
            std::cout << std::endl;
            ONIAK::write_file<double, double>(dump_file, hbe_estimators);
        }
        //for (auto& val: re_mres) val /= selected_queries.size() * kNumIters;
        for (auto& val: hbs_mres) val /= selected_queries.size() * kNumIters;

        ONIAK::print_one_line(fout, conf.gamma);
        // ONIAK::print_one_line(fout, re_mres);
        ONIAK::print_one_line(fout, hbs_mres);
    }
    return 0;
}
