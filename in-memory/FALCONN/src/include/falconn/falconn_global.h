#ifndef __FALCONN_GLOBAL_H__
#define __FALCONN_GLOBAL_H__

#include <memory>
#include <format>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <type_traits>

#include <Eigen/Dense>
#include <Exception.h>
#include "constexpr.h"

#include "boost/unordered_set.hpp"
#include "boost/unordered_map.hpp"
#include "boost/dynamic_bitset.hpp"
namespace falconn
{

  // Block of used Boost classes
  template <typename T>
  using HashSet = boost::unordered_set<T>;
  template <typename Key, typename T>
  using HashMap = boost::unordered_map<Key, T>;
  // by default unsigned long is 64 bits, here we make it explicit.
  using BoostBitSet = boost::dynamic_bitset<uint64_t>;

  class FalconnError : public std::logic_error
  {
  public:
    FalconnError(const std::string &msg) : logic_error(msg) {}
  };

  namespace global
  {
    std::vector<std::vector<KeyType>> filtered_keys;
  }

  ///
  /// Type for dense points / vectors. The coordinate type can be either float
  /// or double (i.e., use DenseVector<float> or DenseVector<double>). In most
  /// cases, float (single precision) should be sufficient.
  ///
  template <typename CoordinateType>
  using DenseVector =
      Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

  template <typename CoordinateType>
  using DenseMatrix = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  typedef std::pair<DenseVector<CoordinateType>, DenseVector<CoordinateType>> VectorPairType;
  typedef std::pair<const DenseVector<CoordinateType> &, const DenseVector<CoordinateType> &>
      VectorPairRefType;
  typedef DenseVector<CoordinateType> VectorType;
  typedef DenseMatrix<CoordinateType> MatrixType;

  /* CONFIG: change point type if type of data is changed.*/
  using PointType = std::conditional_t<kSelfKroneckerData, VectorType, VectorPairType>;

  using PointSet = std::vector<PointType>;

  ///
  /// Type for sparse points / vectors. The coordinate type can be either float
  /// or double (i.e., use SparseVector<float> or SparseVector<double>). In most
  /// cases, float (single precision) should be sufficient.
  ///
  /// The elements of the vector must be sorted by the index (the first
  /// component of the pair).
  ///
  /// Optionally, you can also change the type of the coordinate indices. This
  /// might be useful if you have indices that fit into an int16_t and you want
  /// to save memory.
  ///
  template <typename CoordinateType, typename IndexType = int32_t>
  using SparseVector = std::vector<std::pair<IndexType, CoordinateType>>;

  ///
  /// Data structure for point query statistics
  ///
  struct QueryStatistics
  {
    ///
    /// Average total query time
    ///
    double average_total_query_time = 0.0;
    ///
    /// Average hashing time
    ///
    double average_lsh_time = 0.0;

    double average_multiprobe_time = 0.0;
    ///
    /// Average hash table retrieval time
    ///
    double average_hash_table_time = 0.0;

    double average_sketches_time = 0.0;
    ///
    /// Average time for computing distances
    ///
    double average_distance_time = 0.0;
    ///
    /// Average number of candidates
    ///
    double average_num_candidates = 0;
    ///
    /// Average number of *unique* candidates
    ///
    double average_num_unique_candidates = 0;

    double average_num_filtered_candidates = 0;
    ///
    /// Number of queries the statistics were computed over
    ///
    int_fast64_t num_queries = 0;

    // TODO: move these to internal helper functions?
    void convert_to_totals()
    {
      average_total_query_time *= num_queries;
      average_lsh_time *= num_queries;
      average_hash_table_time *= num_queries;
      average_sketches_time *= num_queries;
      average_distance_time *= num_queries;
      average_num_candidates *= num_queries;
      average_num_unique_candidates *= num_queries;
      average_num_filtered_candidates *= num_queries;
      average_multiprobe_time *= num_queries;
    }

    void compute_averages()
    {
      if (num_queries > 0)
      {
        average_total_query_time /= num_queries;
        average_lsh_time /= num_queries;
        average_hash_table_time /= num_queries;
        average_sketches_time /= num_queries;
        average_distance_time /= num_queries;
        average_multiprobe_time /= num_queries;
        average_num_candidates /= num_queries;
        average_num_unique_candidates /= num_queries;
        average_num_filtered_candidates /= num_queries;
      }
    }

    void add_totals(const QueryStatistics &other)
    {
      average_total_query_time += other.average_total_query_time;
      average_lsh_time += other.average_lsh_time;
      average_hash_table_time += other.average_hash_table_time;
      average_sketches_time += other.average_sketches_time;
      average_distance_time += other.average_distance_time;
      average_num_candidates += other.average_num_candidates;
      average_num_unique_candidates += other.average_num_unique_candidates;
      average_num_filtered_candidates += other.average_num_filtered_candidates;
      average_multiprobe_time += num_queries;
      num_queries += other.num_queries;
    }

    void reset()
    {
      average_total_query_time = 0.0;
      average_lsh_time = 0.0;
      average_hash_table_time = 0.0;
      average_sketches_time = 0.0;
      average_distance_time = 0.0;
      average_num_candidates = 0.0;
      average_num_unique_candidates = 0.0;
      average_multiprobe_time = 0.0;
      average_num_filtered_candidates = 0.0;
      num_queries = 0;
    }

    void dump(std::ostream &stream)
    {
      stream << "Average total query time: " << average_total_query_time << std::endl;
      stream << "Average LSH time: " << average_lsh_time << std::endl;
      stream << "Average multi-probe time: " << average_multiprobe_time << std::endl;
      stream << "Average hash table time: " << average_hash_table_time << std::endl;
      stream << "Average sketches time: " << average_sketches_time << std::endl;
      stream << "Average distance time: " << average_distance_time << std::endl;
      stream << "Average # candidates: " << average_num_candidates << std::endl;
      stream << "Average # unique candidates: " << average_num_unique_candidates << std::endl;
      stream << "Average # filtered candidates: " << average_num_filtered_candidates << std::endl;
      stream << "# queries: " << num_queries << std::endl;
    }
  };

  ///
  /// A struct for wrapping point data stored in a single dense data array. The
  /// coordinate order is assumed to be point-by-point (row major), i.e., the
  /// first dimension coordinates belong to the first point and there are
  /// num_points points in total.
  ///
  template <typename CoordinateType>
  struct PlainArrayPointSet
  {
    const CoordinateType *data;
    int_fast32_t num_points;
    int_fast32_t dimension;
  };

  template <typename T>
  void ptrs_sanity_check(const std::unique_ptr<T> &p)
  {
    std::string message = "Error! Pointer not correctly initialized! Type: ";
    if (!p)
    {
      message += typeid(T).name();
      throw FalconnError(message);
    }
  }

  template <typename CoordinateType>
  DenseMatrix<CoordinateType> read_eigen_matrix(std::ifstream &fin, int rows, int cols)
  {
    DenseMatrix<CoordinateType> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr)
    {
      for (int cc = 0; cc < cols; ++cc)
      {
        fin >> result(rr, cc);
      }
    }
    return result;
  }

  template <typename CoordinateType>
  DenseMatrix<CoordinateType> read_one_matrix(std::ifstream &fin, int rows, int cols)
  {
    static_assert(sizeof(CoordinateType) == 4);

    char buff[4];
    DenseMatrix<CoordinateType> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr)
    {
      fin.read(buff, 4);
      for (int cc = 0; cc < cols; ++cc)
      {
        fin.read(reinterpret_cast<char *>(&result(rr, cc)), 4);
      }
    }
    return result;
  }

  // used for reading ground truth
  // The returned results are sorted for each query in order for computing set instersection.
  typedef std::vector<std::vector<int>> IntegerMatrix;
  IntegerMatrix read_ground_truth(std::ifstream &fin, int K, int qn)
  {
    int sz;
    IntegerMatrix result;
    while (fin.read(reinterpret_cast<char *>(&sz), 4))
    {
      std::vector<int> cur_gt(K);
      fin.read(reinterpret_cast<char *>(cur_gt.data()), 4 * K);
      assert(sz >= K && "Ground truth must contain at least K nearest neighbors.");
      fin.seekg((sz - K) * 4, std::ios::cur); // skip the remaining neighbors for this query
      std::sort(cur_gt.begin(), cur_gt.end());
      result.push_back(std::move(cur_gt));
    }
    assert(result.size() == static_cast<size_t>(qn));
    return result;
  }

  template <typename CoordinateType>
  DenseVector<CoordinateType> read_vector(std::ifstream &fin)
  {
    static_assert(sizeof(CoordinateType) == 4);

    int sz;
    fin.read(reinterpret_cast<char *>(&sz), 4);
    DenseVector<CoordinateType> result(sz);
    fin.read(reinterpret_cast<char *>(&result(0)), 4 * sz);

    // TODO: test this function
    // for (int cc = 0; cc < sz; ++cc) {
    //      fin.read(reinterpret_cast<char*> (&result(cc)), 4);
    //  }

    return result;
  }

  class NearestNeighborQueryError : public FalconnError
  {
  public:
    NearestNeighborQueryError(std::string msg) : FalconnError(msg) {}
  };

  class LSHFunctionError : public FalconnError
  {
  public:
    LSHFunctionError(std::string msg) : FalconnError(msg) {}
  };

  struct FalconnCandidateType
  {
    int_fast64_t id;
    int_fast32_t table, bucket_order;
    CoordinateType q2d_distance;
  };

  // Type of probing list in multi-probe
  using FalconnProbingListType = std::vector<std::vector<int>>;

  ///
  /// Supported LSH families.
  ///
  enum class LSHFamily
  {
    Unknown = 0,

    ///
    /// The hyperplane hash proposed in
    ///
    /// "Similarity estimation techniques from rounding algorithms"
    /// Moses S. Charikar
    /// STOC 2002
    ///
    Hyperplane = 1,

    ///
    /// The cross polytope hash first proposed in
    ///
    /// "Spherical LSH for Approximate Nearest Neighbor Search on Unit
    //   Hypersphere",
    /// Kengo Terasawa, Yuzuru Tanaka
    /// WADS 2007
    ///
    /// Our implementation uses the algorithmic improvements described in
    ///
    /// "Practical and Optimal LSH for Angular Distance",
    /// Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig
    ///   Schmidt
    /// NIPS 2015
    ///
    CrossPolytope = 2,
    Gaussian = 3
  };

  enum class GaussianFunctionType
  {
    Unknown = 0,
    Cauchy = 1,
    L1Precompute = 2,
    L1DyadicSim = 3
  };

  enum class NNQueryVariant
  {
    Unknown = 0,
    TopK = 1,
    RadiusR = 2
  };

  ///
  /// Supported distance functions.
  ///
  /// Note that we use distance functions only to filter the candidates in
  /// find_nearest_neighbor, find_k_nearest_neighbors, and find_near_neighbors.
  //  For only returning all the candidates, the distance function is irrelevant.
  ///
  enum class DistanceFunction
  {
    Unknown = 0,

    ///
    /// The distance between p and q is -<p, q>. For unit vectors p and q,
    /// this means that the nearest neighbor to q has the smallest angle with q.
    ///
    NegativeInnerProduct = 1,

    ///
    /// The distance is the **squared** Euclidean distance (same order as the
    /// actual Euclidean distance / l2-distance).
    ///
    EuclideanSquared = 2,

    // The distance is L1 Manhattan distance
    L1Norm = 3,
    Euclidean = 4
  };

  enum class MultiProbeType
  {
    Unknown = 0,
    Legacy = 1, // use multiprobe that is included in lsh functions
    Customized = 2,
    Precomputed = 3
  };

  ///
  /// Supported low-level storage hash tables.
  ///
  enum class StorageHashTable
  {
    Unknown = 0,
    ///
    /// The set of points whose hash is equal to a given one is retrieved using
    /// "naive" buckets. One table takes space O(#points + #bins).
    ///
    FlatHashTable = 1,
    ///
    /// The same as FlatHashTable, but everything is packed using as few bits as
    /// possible. This option is recommended unless the number of bins is much
    /// larger than the number of points (in which case we recommend to use
    /// LinearProbingHashTable).
    ///
    BitPackedFlatHashTable = 2,
    ///
    /// Here, std::unordered_map is used. One table takes space O(#points), but
    /// the leading constant is much higher than that of bucket-based approaches.
    ///
    STLHashTable = 3,
    ///
    /// The same as STLHashTable, but the custom implementation based on
    /// *linear probing* is used. This option is recommended if the number of
    /// bins is much higher than the number of points.
    ///
    LinearProbingHashTable = 4
  };

  enum class FalconnQueryMode
  {
    Unknown = 0,
    // Compute K nearest neighbors, return recall and query time
    KNNWithTime = 1,
    // Print candidates, used for tuning or estimating
    PrintCandidates = 2,
    // Computes KNN recalls by comparing with ground truth, used for tuning
    KNNRecall = 3,
    // Print hashed vectors of queries
    PrintHashedQueries = 4,
    // Print probing sequences of queries
    PrintProbingSequence = 5,
    // Print hashed vectors of data
    PrintHashedData = 6,
    // Print duplicated candidates
    PrintDuplicateCandidates = 7,
    // Print time measurements for probing candidates
    PrintTimeMeasurements = 8,
    // Print precomputed multi-probe sequence
    PrintPrecomputedMPSequence = 9,
    // Print Random variables in Hash function.
    PrintHashFunction = 10,
    // Print Raw Hashed Data
    PrintRawHashData = 11,
    // Measure KNN Time (including deduplication and distance calculation)
    MeasureKNNTime = 12,
    // Print ground truth of QDDE
    QDDEGroundTruth = 13,
    CandidateNum = 19,
    // Future CanDE tasks will be moved under this category
    KNNWithCanDE = 20,
  };

  const char *CANDE_TASK_NAMES[] = {"unknown", "KDE", "QDDE", "recall"};
  enum class CanDETask
  {
    kUnknown = 0,
    kKDE = 1,
    kQDDE = 2,
    kRecall = 3
  };
  std::string_view cande_task_name(CanDETask task)
  {
    return CANDE_TASK_NAMES[static_cast<int>(task)];
  }

  // kPrecompute: use precomputed collision probability
  // kAssociative: use CanDE infer implemented by associated memory
  // kHash: use CanDE infer implemented by hash table
  const char *CANDE_ALGO_NAMES[] =
      {"unknown", "precompute", "associative", "hash-table", "weighted recall",
       "CP adjusted", "resample"};
  enum class CanDEAlgoType
  {
    kUnknown = 0,
    kPrecompute = 1,
    kAssociative = 2,
    kHash = 3,
    kWeightedRecall = 4,
    kCPAdjusted = 5,
    kResample = 6
  };
  std::string_view cande_algo_name(CanDEAlgoType algo)
  {
    return CANDE_ALGO_NAMES[static_cast<int>(algo)];
  }

  // Defines which fields of candidates are printed.
  struct CandidatePrintingMode
  {
    // Print candidate id
    bool print_id = true;
    // Print query-to-data distance
    bool print_distance = false;
    // Print LSH table id (useful in parameter tuning)
    bool print_table = false;
    // Print the order of bucket in multi-probe sequence
    bool print_bucket_order = false;
  };

  struct CompositeHashTableParameters
  {
    int_fast32_t k = -1;
    ///
    /// Number of hash tables. Required for all the hash families.
    ///
    int_fast32_t l = -1;
    float bucket_width = -1.0;
    CoordinateType partition_lower = -1.0;
    CoordinateType partition_upper = -1.0;

    auto operator<=>(const CompositeHashTableParameters &) const = default;
  };

  struct FalconnRange
  {
    double start, end, step;
    int count;
    FalconnRange(double stt, double ed, double stp, int cnt = -1) : start(stt), end(ed), step(stp), count(cnt) {}

    int bin_translate(double distance) const
    {
      distance = (distance > end) ? end : distance;
      int idx = std::floor((distance - start) / step) + 1;
      idx = (idx < 0) ? 0 : idx;
      return idx;
    }

    // returns the upper bound of the bin
    double upper_bound(int idx) const
    {
      return start + idx * step;
    }

    double mid_point(int idx) const
    {
      return start + (idx - 0.5) * step;
    }

    int num_bins() const { return bin_translate(end) + 1; }
  };

  ///
  /// Contains the parameters for constructing a LSH table wrapper. Not all fields
  /// are necessary for all types of LSH tables.
  ///
  struct LSHConstructionParameters
  {
    // Required parameters
    ///
    /// Dimension of the points. Required for all the hash families.
    ///
    int_fast32_t dimension = -1;
    ///
    /// Hash family. Required for all the hash families.
    ///
    // LSHFamily lsh_family = LSHFamily::Unknown;
    ///
    /// Distance function. Required for all the hash families.
    ///
    // DistanceFunction distance_function = DistanceFunction::Unknown;
    ///
    /// Number of hash function per table. Required for all the hash families.
    ///

    /// Number to jump for random walk
    int step = -1;
    ///
    /// Low-level storage hash table.
    ///
    // StorageHashTable storage_hash_table = StorageHashTable::Unknown;

    // MultiProbeType multi_probe = MultiProbeType::Unknown;

    // GaussianFunctionType gauss_type = GaussianFunctionType::Unknown;
    ///
    /// Number of threads used to set up the hash table.
    /// Zero indicates that FALCONN should use the maximum number of available
    /// hardware threads (or 1 if this number cannot be determined).
    /// The number of threads used is always at most the number of tables l.
    ///
    int_fast32_t num_setup_threads = -1;
    ///
    /// Randomness seed.
    ///
    uint64_t seed = 409556018;

    // Optional parameters
    ///
    /// Dimension of the last of the k cross-polytopes. Required
    /// only for the cross-polytope hash.
    ///
    // int_fast32_t last_cp_dimension = -1;
    ///
    /// Number of pseudo-random rotations. Required only for the
    /// cross-polytope hash.
    ///
    /// For sparse data, it is recommended to set num_rotations to 2.
    /// For sufficiently dense data, 1 rotation usually suffices.
    ///
    int_fast32_t num_rotations = -1;
    ///
    /// Intermediate dimension for feature hashing of sparse data. Ignored for
    /// the hyperplane hash. A smaller feature hashing dimension leads to faster
    /// hash computations, but the quality of the hash also degrades.
    /// The value -1 indicates that no feature hashing is performed.
    ///
    // int_fast32_t feature_hashing_dimension = -1;
    //  bucket width of gaussian hash function

    // id width of gaussian function
    int_fast32_t bucket_id_width = -1;
    // Universe of database used for L1
    int_fast32_t universe = 1;
    // bit width of the underlying hash table
    int_fast32_t hash_table_width = 1;
    // bit width of hash table for hash-implemented CanDE
    size_t cande_table_size = 1;
    // whether to save LSH index to avoid future recomputations
    bool save_index = true;
    // name of index file
    std::filesystem::path index_filename = "";
    std::string summary_path = "";
    // path for dumping performance measures for the indexing stage
    std::filesystem::path index_record_path = "";
    std::string eigen_filename = "", result_filename = "", candidate_filename = "", rowid_filename = "";
    // name of ground truth file
    std::string gnd_filename = "";
    std::string data_filename = "", query_filename = "", kernel_filename = "", prob_filename = "";
    std::string distance_filename = "", query_hash_filename = "", probing_sequence_file = "";
    std::string recall_p_filename = "", mp_prob_filename = "", hash_func_filename = "";
    // name of result files
    std::string histogram_filename = "", knn_filename = "", kde_filename = "", middle_result_filename = "";
    std::string accuracy_filename = "";
    std::string accuracy_binary_filename = "";
    int rotation_dim = -1;
    int num_prehash_filters = 0;
    double prefilter_ratio = 1.0;
    double avg_radius = 0.0;
    double tau = 0.0; // we do not count accuracy when KDE value is less than this
    double target_error = 0.0, error_range = 0.0;
    std::string dataset = "";
    std::vector<double> radius_R = {};
    DoubleTriplet histogram_bins = {0.0, 0.0, 0.0};
    std::vector<FalconnRange> bins_vector = {};
    double bayesian_prior = 25.0;
    // loads precomputed disances, speeds up debugging and tuning
    bool load_distance = false;
    bool hypergeometric_bayesian = false;

    int_fast32_t num_partitions = 0;
    std::vector<CompositeHashTableParameters> hash_table_params;
    int num_hash_funcs = 0;

    bool second_step = false, fast_rotation = false;
    bool single_kernel = false;
    // Use a single kernel for all queries
    bool transformed_queries = false;
    // If this flag is true, the input queries are already transformed
    // Otherwise, the input query is transformed in this program.
    int dim_Arows = 0, dim_Acols = 0;
    bool allow_overwrite = false;
    bool compute_gound_truth = false;
    int num_neighbors = 0, num_points = 0, num_queries = 0, raw_dimension = 0;
    int probes_per_table = 0, num_experiments = 0, max_attempts = 0;

    std::vector<double> gamma = {}; // used for kde
    std::vector<int> num_neighbors_list = {};

    FalconnQueryMode query_mode = FalconnQueryMode::Unknown;
    NNQueryVariant query_variant = NNQueryVariant::TopK;
    CandidatePrintingMode printing_mode = CandidatePrintingMode();

    std::unique_ptr<std::seed_seq> seeding_sequence = nullptr;

    CanDETask cande_task = CanDETask::kUnknown;
    std::vector<CanDEAlgoType> cande_algos = {};
  };

  // global variable for configuration
  inline LSHConstructionParameters falconn_config;

} // namespace falconn

#endif
