#ifndef STYLIZATION_H
#define STYLIZATION_H

#include "mesh.h"
#include <unordered_map>
#include <Eigen/Sparse>

class Stylization
{
public:
    Stylization(HalfEdgeMesh* m);
    void transform(double lambda, bool printObj);
    void fastTransform(double lambda, int m, bool printObj);
private:
    static const Eigen::IOFormat defaultFmt;
    HalfEdgeMesh* _mesh;
    double _lambda;
    std::vector<double> _rho;
    std::vector<double> _a;
    std::vector<Eigen::Vector3d> _z;
    std::vector<Eigen::Vector3d> _u;
    std::vector<Eigen::Vector3d> _V;
    std::vector<Eigen::Vector3d> _deformedV;
    std::vector<Eigen::Vector3d> _n;
    std::vector<Eigen::Matrix3Xd> _D;
    std::vector<Eigen::MatrixX3d> _deformedD;
    std::vector<Eigen::MatrixXd> _W;
    std::vector<Eigen::Matrix3d> _R;
    std::vector<Eigen::Vector3d> _deltaZ;
    std::unordered_map<Vertex*, size_t> _index;
    Eigen::SparseMatrix<double> _sysMatrix;
    std::vector<double> _r;
    std::vector<double> _s;
    void buildUnitAreaWeightedNormal();
    void buildBarycentricArea();
    void buildD();
    void buildW();
    void updateW();
    void updateDeformedV();
    void updateDeformedD();
    void buildSysMat();
    void reset(double lambda);
    void updateM(std::vector<Eigen::Matrix3d> &ret);
    void updateR();
    void updateZ();
    void updateU(std::vector<Eigen::Vector3d> &ret);
    void updateRho(const std::vector<Eigen::Vector3d> &tempU);
    void evalTerminate();
    void evalObjective(std::ofstream &out);
    void localStep();
    void globalStep();
    void validate();
};

#endif // STYLIZATION_H
