#include "falconn/falconn_global.h"
#include <falconn/lsh_nn_table.h>
#include "falconn/config.h"
#include "falconn/fileio.h"
#include "falconn/peakrss.h"
#include "falconn/query_mode.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include "VectorUtils.hpp"

#include <AnnResultWriter.hpp>
#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>
#include <cstdio>

using namespace StringUtils;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using namespace VectorUtils;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::get_default_parameters;
using falconn::GaussianFunctionType;
using falconn::MultiProbeType;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::read_config;
using falconn::read_eigen_matrix;
using falconn::DenseMatrix;
using falconn::CoordinateType;

using Point = falconn::PointType;

// print parameters to stdout
void show_params(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    char *name = va_arg(args, char *);
    if (*fmt == 'i') {
      int val = va_arg(args, int);
      printf("%s: %d\n", name, val);
    } else if (*fmt == 'c') {
      int val = va_arg(args, int);
      printf("%s: \'%c\'\n", name, val);
    } else if (*fmt == 'f') {
      double val = va_arg(args, double);
      printf("%s: %f\n", name, val);
    } else if (*fmt == 's') {
      char *val = va_arg(args, char *);
      printf("%s: \"%s\"\n", name, val);
    } else {
      NPP_ERROR_MSG("Unsupported format");
    }
    ++fmt;
  }

  va_end(args);
}

void usage() {
  printf("Falconn\n");
  printf("Options\n");
  printf("-cf {string}     \trequired \tconfig file path\n");
  printf("\n");
}

std::unique_ptr<LSHNearestNeighborTable<Point>>
indexing(const LSHConstructionParameters &params,  // config filename
         std::vector<Point> &train
        ) { 
 
  std::filesystem::path perf_filename = std::format("n{}-d{}-l{}-m{}.txt",
                                          params.num_points,
                                          params.dimension,
                                          params.hash_table_params[0].l,
                                          params.hash_table_params[0].k);
  
  perf_filename = perf_filename / params.index_record_path;

  AnnResultWriter writer(perf_filename, params.allow_overwrite);
  writer.writeRow(
      "s",
      "dsname,#n,#dim,#hashes,#functions,index_size(bytes),construction_time(us)");
  const char *fmt = "siiiiif";

  HighResolutionTimer timer;
  timer.restart();
  auto table = construct_table<Point>(train, params);
  auto e = timer.elapsed();

  auto isz = getPeakRSS();

  writer.writeRow(fmt, params.data_filename.c_str(), params.num_points, params.dimension, params.hash_table_params[0].l, params.hash_table_params[0].k, isz, e);

  return table;
}

/*
 * Get the index of next unblank char from a string.
 */
int GetNextChar(char *str) {
  int rtn = 0;

  // Jump over all blanks
  while (str[rtn] == ' ') {
    rtn++;
  }

  return rtn;
}

/*
 * Get next word from a string.
 */
void GetNextWord(char *str, char *word) {
  // Jump over all blanks
  while (*str == ' ') {
    str++;
  }

  while (*str != ' ' && *str != '\0') {
    *word = *str;
    str++;
    word++;
  }

  *word = '\0';
}

int main(int argc, char **argv) {
  int cnt = 1;
  bool failed = false;
  char *arg;
  int i;
  char para[10];

  char conf[200] = "";

  std::string err_msg;
  while (cnt < argc && !failed) {
    arg = argv[cnt++];
    if (cnt == argc) {
      failed = true;
      break;
    }

    i = GetNextChar(arg);
    if (arg[i] != '-') {
      failed = true;
      err_msg = "Wrong format!";
      break;
    }

    GetNextWord(arg + i + 1, para);

    arg = argv[cnt++];

    if (strcmp(para, "cf") == 0) {
      GetNextWord(arg, conf);
    } else {
      failed = true;
      fprintf(stderr, "Unknown option -%s!\n\n", para);
    }
  }

  if (failed) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__, err_msg.c_str());
    usage();
    return EXIT_FAILURE;
  }

    LSHConstructionParameters params = read_config(conf);
    
    #ifndef DISABLE_VERBOSE
  printf("=====================================================\n");
  show_params("iiiiiiifssssi", "# of points", params.num_points, "dimension",
             params.dimension, "# of queries", params.num_queries, "# of hash tables", params.hash_table_params[0].l,
             "# of hash functions", params.hash_table_params[0].k, "# of probes per table", params.probes_per_table, "k", params.num_neighbors,
             "bucket width",params.hash_table_params[0].bucket_width, 
             "dataset filename", params.data_filename.c_str(), "configuration path",conf, "result filename",
             params.result_filename.c_str(), "ground truth filename", params.gnd_filename.c_str(),
             "falconn query mode", static_cast<int>(params.query_mode));
  printf("=====================================================\n");
#endif

    falconn::falconn_config = std::move(params);
    falconn::falconn_query_mode();

  return EXIT_SUCCESS;
}
