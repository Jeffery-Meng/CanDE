#include "../fvecs.hpp"
#include <iostream>

using namespace std;

int main() {
    string datafile = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/hepmass/hepmass_data.fvecs";
    string candifile = "";
    string gtfile = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/indices/hepmass_full_gnd.ivecs";
    auto data_in = open_binary(datafile);
    auto data = read_one_matrix(data_in, 10000000, 27);

    cout << (Eigen::VectorXf) data.row(0) << endl;
    cout << (Eigen::VectorXf) data.row(1) << endl;
    cout << (Eigen::VectorXf) data.row(2000000) << endl;
    cout << (Eigen::VectorXf) data.row(9999999) << endl;

    auto gt_in = open_binary(gtfile);
    auto gt = read_ground_truth(gt_in, 50, 100);
    cout << gt[0] << endl;
    cout << gt[1] << endl;
    cout << gt[49] << endl;
    return 0;
}