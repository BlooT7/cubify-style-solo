#include "stylization.h"

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <fstream>

using namespace Eigen;
using namespace std;

#define DEFAULT_RHO 1e-4
#define MU 10.
#define TAU_INCR 2.
#define TAU_DECR 2.
#define EPSILON_ABS 1e-5
#define EPSILON_REL 1e-3

const IOFormat Stylization::defaultFmt = IOFormat(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

Stylization::Stylization(HalfEdgeMesh* mesh)
{
    this->_mesh = mesh;
    const size_t n = mesh->getNumVertices();
    for (size_t i = 0; i < n; ++i) {
        this->_index.insert(make_pair(this->_mesh->getVertex(i), i));
    }
}

inline Vector3d getAreaWeightedNormal(const HalfEdge* e)
{
    const Vector3d pt0 = e->vertex->xyz;
    const Vector3d pt1 = e->next->vertex->xyz;
    const Vector3d pt2 = e->next->next->vertex->xyz;
    // The half of norm of the cross product would be the area.
    // Because we only take the area as a weight so we don't divide it by two.
    // And, a normalized cross product would be the normal vector.
    return (pt1 - pt0).cross(pt2 - pt0);
}

void Stylization::buildUnitAreaWeightedNormal()
{
    this->_n.resize(this->_mesh->getNumVertices());
    // This is a buffer for storing calculated area-weighted normal vectors based on face pointer.
    // This is each face will connect to three vertices, and we can eliminate the repetitive calculation.
    unordered_map<Face*, Vector3d> face2Normal;
    function<Vector3d(HalfEdge*)> accumulateAreaWeightedNormal = [&face2Normal](HalfEdge* e) -> Vector3d {
        Face* f = e->face;
        auto iter = face2Normal.find(f);
        if (iter == face2Normal.end()) {
            Vector3d n = getAreaWeightedNormal(e);
            face2Normal.insert(make_pair(f, n));
            return n;
        }
        return iter->second;
    };
    function<Vector3d(const vector<Vector3d>&)> normalization = [](const vector<Vector3d> &ls) -> Vector3d {
        Vector3d n = Vector3d::Zero();
        for (Vector3d temp: ls) {
            n += temp;
        }
        // Normalize it so we can get a unit normal.
        n.normalize();
        return n;
    };
    // Populate the normalized n vector for each vertex.
    for (size_t i = 0; i < this->_n.size(); ++i) {
        this->_n[i] = this->_mesh->traverseFan(this->_mesh->getVertex(i), accumulateAreaWeightedNormal, normalization);
    }
}

inline double cot(const Vector3d &x, const Vector3d &y)
{
    // By definition of sin/cos/cot.
    return x.dot(y) / x.cross(y).norm();
}

inline double getCotWeight(const HalfEdge* e)
{
    const Vector3d pt0 = e->next->next->vertex->xyz;
    const Vector3d pt1 = e->next->vertex->xyz;
    const Vector3d pt2 = e->vertex->xyz;
    return cot(pt1 - pt0, pt2 - pt0);
}

void Stylization::buildW()
{
    this->_W.resize(this->_mesh->getNumVertices());
    function<double(HalfEdge*)> weightCollector = [](HalfEdge* e) -> double {
        // Calculate cotangent weight for each edge in the fan structure.
        return (abs(getCotWeight(e)) + abs(getCotWeight(e->twin))) * .5;
    };
    function<MatrixXd(const vector<double>&)> diagBuilder = [](const vector<double> &ls) -> MatrixXd {
        // The extra row/column is for rho which should be updated for each iteration.
        // Other entries should remain unchanged.
        MatrixXd m(ls.size() + 1, ls.size() + 1);
        m.setZero();
        // Populate the weights.
        for (size_t i = 0; i < ls.size(); ++i) {
            m(i, i) = ls[i];
        }
        return m;
    };
    for (size_t i = 0; i < this->_W.size(); ++i) {
        this->_W[i] = this->_mesh->traverseFan(this->_mesh->getVertex(i), weightCollector, diagBuilder);
        const int n = this->_W[i].rows() - 1;
        this->_W[i](n, n) = this->_rho[i];
    }
}

void Stylization::updateW()
{
    // Update the Rho entry in the diagonal matrices.
    for (size_t i = 0; i < this->_W.size(); ++i) {
        const int n = this->_W[i].rows() - 1;
        this->_W[i](n, n) = this->_rho[i];
    }
}

void Stylization::buildD()
{
    this->_D.resize(this->_mesh->getNumVertices());
    function<Vector3d(HalfEdge*)> vecCollector = [](HalfEdge* e) -> Vector3d {
        return e->next->vertex->xyz - e->vertex->xyz;
    };
    function<Matrix3Xd(const vector<Vector3d>&)> DBuilder = [](const vector<Vector3d> &ls) -> Matrix3Xd {
        // The extra column is for unit weighted normal.
        // This matrix should not be updated during iterations.
        Matrix3Xd D = MatrixXd(3, ls.size() + 1);
        for (size_t i = 0; i < ls.size(); ++i) {
            D.col(i) << ls[i];
        }
        return D;
    };
    for (size_t i = 0; i < this->_D.size(); ++i) {
        // Assign D matrix.
        this->_D[i] = this->_mesh->traverseFan(this->_mesh->getVertex(i), vecCollector, DBuilder);
        // Assign the processed n vector.
        this->_D[i].rightCols(1) << this->_n[i];
    }
}

void Stylization::updateDeformedV()
{
    // The V vector remains unchanged, and the updated V would be stored in deformedV field.
    this->_mesh->updateVertices(this->_deformedV);
}

void Stylization::updateDeformedD()
{
    function<Vector3d(HalfEdge*)> vecCollector = [](HalfEdge* e) -> Vector3d {
        return e->next->vertex->xyz - e->vertex->xyz;
    };
    function<MatrixX3d(const vector<Vector3d>&)> DBuilder = [](const vector<Vector3d> &ls) -> MatrixX3d {
        // The extra column is for unit weighted normal.
        // This matrix should not be updated during iterations.
        MatrixX3d D(ls.size() + 1, 3);
        D.setZero();
        for (size_t i = 0; i < ls.size(); ++i) {
            D.row(i) << ls[i].transpose();
        }
        return D;
    };
    for (size_t i = 0; i < this->_deformedD.size(); ++i) {
        // Assign deformed D matrix.
        this->_deformedD[i] = this->_mesh->traverseFan(this->_mesh->getVertex(i), vecCollector, DBuilder);
        // Assign the predicted vector.
        const Vector3d temp = this->_z[i] - this->_u[i];
        this->_deformedD[i].bottomRows(1) << temp.transpose();
    }
}

inline double getArea(const HalfEdge* e)
{
    // The half of cross product norm is triangle area.
    // For convenience, the called function returns only the cross product itself without scaling.
    return getAreaWeightedNormal(e).norm();
}

void Stylization::buildBarycentricArea()
{
    this->_a.resize(this->_mesh->getNumVertices());
    unordered_map<Face*, double> face2Area;
    function<double(HalfEdge*)> areaCollector = [&face2Area](HalfEdge* e) -> double {
        Face* f = e->face;
        auto iter = face2Area.find(f);
        if (iter == face2Area.end()) {
            double a = getArea(e);
            face2Area.insert(make_pair(f, a));
            return a;
        }
        return iter->second;
    };
    function<double(const vector<double>&)> accumulator = [](const vector<double> &ls) -> double {
        // The area has not been halved for triangle area calculating yet.
        double acc = 0;
        for (double f: ls) {
            acc += f;
        }
        // We need to divide it by two for triangle area and then divide it by three for barycentric area.
        return acc / 6.;
    };
    for (size_t i = 0; i < this->_a.size(); ++i) {
        this->_a[i] = this->_mesh->traverseFan(this->_mesh->getVertex(i), areaCollector, accumulator);
    }
}

void Stylization::buildSysMat()
{
    const size_t n = this->_V.size();
    // Resize the system matrix.
    this->_sysMatrix = SparseMatrix<double>(3 * n, 3 * n);
    function<double(const vector<double>&)> accumulator = [](const vector<double> &ls) -> double {
        double acc = 0.;
        for (double f: ls) {
            acc += f;
        }
        return acc;
    };
    vector<Triplet<double>> ijv;
    for (size_t i = 0; i < n; ++i) {
        size_t j = 0;
        function<double(HalfEdge*)> weightCollector = [this, &i, &j, &ijv](HalfEdge* e) -> double {
            const double w = this->_W[i](j, j);
            auto iter = this->_index.find(e->next->vertex);
            const size_t ind = iter->second * 3;
            ijv.push_back(Triplet<double>(3*i, ind, w));
            ijv.push_back(Triplet<double>(3*i+1, ind+1, w));
            ijv.push_back(Triplet<double>(3*i+2, ind+2, w));
            ++j;
            return w;
        };
        const double w = -this->_mesh->traverseFan(this->_mesh->getVertex(i), weightCollector, accumulator);
        ijv.push_back(Triplet<double>(3*i, 3*i, w));
        ijv.push_back(Triplet<double>(3*i+1, 3*i+1, w));
        ijv.push_back(Triplet<double>(3*i+2, 3*i+2, w));
    }
    this->_sysMatrix.setFromTriplets(ijv.begin(), ijv.end());
}

void Stylization::reset(double lambda)
{
    this->_lambda = lambda;
    // Initialize rho scalars.
    this->_rho.resize(this->_mesh->getNumVertices());
    for (size_t i = 0; i < this->_rho.size(); ++i) {
        this->_rho[i] = DEFAULT_RHO;
    }
    // Populate a scalars for each vertex.
    this->buildBarycentricArea();
    cout << "Calculate barycentric areas (a_i) for all vertices." << endl;
    // Populate V vector for each vertex.
    // Initialize updated V vector for each vertex.
    this->_V.resize(this->_mesh->getNumVertices());
    this->_deformedV.resize(this->_mesh->getNumVertices());
    // Also, we need to initialize z vectors and u vectors.
    this->_z.resize(this->_mesh->getNumVertices());
    this->_u.resize(this->_mesh->getNumVertices());
    for (size_t i = 0; i < this->_V.size(); ++i) {
        this->_V[i] = this->_mesh->getVertex(i)->xyz;
        this->_deformedV[i] = this->_V[i];
        this->_z[i] = Vector3d::Zero();
        this->_u[i] = Vector3d::Zero();
    }
    cout << "Stores positions of all vertices." << endl;
    cout << "Initialize all buffered deformed positions of all vertices." << endl;
    cout << "Initialize second variables (rotated areas weighted normal vectors)." << endl;
    // Populate R matrix for each vertex.
    this->_R.resize(this->_mesh->getNumVertices());
    for (size_t i = 0; i < this->_R.size(); ++i) {
        this->_R[i] = Matrix3d::Identity();
    }
    cout << "Initialize first variables (rotational matrices)." << endl;
    // Populate n vectors for each vertex.
    this->buildUnitAreaWeightedNormal();
    cout << "Initialize unit area weighted normal vectors of all vertices." << endl;
    // Populate D matrix for each edge.
    this->buildD();
    this->_deformedD.resize(this->_mesh->getNumVertices());
    this->updateDeformedD();
    cout << "Initialize D matrix for each vertex." << endl;
    // Populate cotangent weights for each edge.
    this->buildW();
    cout << "Initialize the cotangent weights matrix for each vertex." << endl;
    // Resize the delta z vector.
    this->_deltaZ.resize(this->_mesh->getNumVertices());
    // Create the system matrix for global step.
    this->buildSysMat();
    cout << "Initialize the system matrix for global steps." << endl; 
    this->_r.resize(this->_mesh->getNumVertices());
    this->_s.resize(this->_mesh->getNumVertices());
}

void Stylization::evalObjective(ofstream &out)
{
    const size_t n = this->_V.size();
    double arap = 0.;
    double cube = 0.;
    for (size_t i = 0; i < n; ++i) {
        MatrixXd X = this->_R[i] * this->_D[i].leftCols(this->_D[i].cols() - 1);
        X -= this->_deformedD[i].transpose().leftCols(this->_deformedD[i].rows() - 1);
        MatrixXd Y = this->_W[i].block(0, 0, this->_W[i].rows() - 1, this->_W[i].cols() - 1);
        arap += .5 * (X * Y * X.transpose()).trace();
        cube += this->_lambda * this->_a[i] * this->_z[i].lpNorm<1>();
    }
    cout << "ARAP: " << arap << "; Cube: " << cube << endl;
    out << arap + cube << endl;
}

void Stylization::transform(double lambda, bool printObj)
{
    // Reset the Parameters
    this->reset(lambda);
    // Test the initialization.
    this->validate();
    // Set the termination condition.
    const double epsScaledAbsPrim = sqrt(6.) * EPSILON_ABS;
    const double epsScaledAbsDual = sqrt(3.) * EPSILON_ABS;
    double epsPri, epsDual;
    int i = 0;
    const int bound = static_cast<int>(this->_V.size() * .95);
    int count;
    ofstream out("outputs/obj.csv");
    do {
        this->localStep();
        this->globalStep();
        // Print the objective.
        if (printObj) {
            this->evalObjective(out);
        }
        // Calculate the tolerances.
        count = 0;
        for (size_t i = 0; i < this->_R.size(); ++i) {
            epsPri = epsScaledAbsPrim + EPSILON_REL * max((this->_R[i] * this->_n[i]).norm(), this->_z[i].norm());
            epsDual = epsScaledAbsDual + EPSILON_REL * abs(this->_n[i].dot(this->_rho[i] * this->_u[i]));
            if (this->_r[i] <= epsPri && this->_s[i] <= epsDual) {
                ++count;
            }
        }
    } while (++i < 1000 && count < bound);
    out.close();
    cout << "Number of iterations: " << i << endl;
}

void Stylization::fastTransform(double lambda, int m, bool printObj)
{
    // Pre-Process.
    const size_t prevN = this->_mesh->getNumVertices();
    cout << "Previous Number of Vertices: " << prevN << endl;
    this->_mesh->simplify(m);
    const size_t n = this->_mesh->getNumVertices();
    this->_index.clear();
    for (size_t i = 0; i < n; ++i) {
        this->_index.insert(make_pair(this->_mesh->getVertex(i), i));
    }
    cout << "Current Number of Vertices: " << this->_mesh->getNumVertices() << endl;
    // Performs transform.
    this->transform(lambda, printObj);
    const int it = max(1, static_cast<int>(floor(log2(static_cast<double>(prevN) / static_cast<double>(this->_mesh->getNumVertices())))) - 1);
    cout << "Number of Iterations for Subdivision: " << it << endl;
    this->_mesh->subdivide(it);
}

void Stylization::updateM(vector<Matrix3d> &ret)
{
    ret.resize(this->_V.size());
    // Update Rho in the second matrix.
    this->updateW();
    // No need for update the first matrix and the second matrix is already updated.
    // Update the third matrix.
    this->updateDeformedD();
    for (size_t i = 0; i < ret.size(); ++i) {
        // Multiply the three matrices.
        ret[i] = this->_D[i] * this->_W[i] * this->_deformedD[i];
    }
}

void Stylization::updateR()
{
    // Update M Matrix
    vector<Matrix3d> M;
    this->updateM(M);
    // Update R Matrix.
    for (size_t i = 0; i < M.size(); ++i) {
        // Calculate updated R based on SVD.
        JacobiSVD<Matrix3d> svd(M[i], ComputeFullV | ComputeFullU);
        Matrix3d U = svd.matrixU();
        Matrix3d R = svd.matrixV() * U.transpose();
        // Check if the det(R) non-negative (if not then shift signs on one column).
        if (R.determinant() < 0.) {
            U.col(2) << -U.col(2);
            R = svd.matrixV() * U.transpose();
        }
        assert(R.determinant() >= 0.);
        this->_R[i] = R;
    }
}

void Stylization::updateZ()
{
    // Update the z vectors.
    for (size_t i = 0; i < this->_z.size(); ++i) {
        const double kappa = this->_lambda * this->_a[i] / this->_rho[i];
        const Vector3d k(kappa, kappa, kappa);
        const Vector3d x = this->_R[i] * this->_n[i] + this->_u[i];
        const Vector3d plus = (x - k).cwiseMax(0.);
        const Vector3d minus = (-x - k).cwiseMax(0.);
        const Vector3d res = plus - minus;
        this->_deltaZ[i] = res - this->_z[i];
        this->_z[i] = res;
    }
}

void Stylization::updateU(vector<Vector3d> &ret)
{
    ret.resize(this->_V.size());
    // Update the u vectors.
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = this->_u[i] + this->_R[i] * this->_n[i] - this->_z[i];
    }
}

void Stylization::updateRho(const vector<Vector3d> &tempU)
{
    // Compute residual vectors and dual residual vectors.
    for (size_t i = 0; i < tempU.size(); ++i) {
        const double r = (this->_R[i] * this->_n[i] - this->_z[i]).norm();
        const double s = (-this->_n[i].transpose() * this->_rho[i] * this->_deltaZ[i]).norm();
        this->_r[i] = r;
        this->_s[i] = s;
        // Update corresponding Rho scalar and fix the value of dual variable u.
        if (r > MU * s) {
            this->_rho[i] *= TAU_INCR;
            // Don't forget to rescale the dual variable.
            this->_u[i] = tempU[i] / TAU_INCR;
        } else if (s > MU * r) {
            this->_rho[i] /= TAU_DECR;
            this->_u[i] = tempU[i] * TAU_DECR;
        } else {
            this->_u[i] = tempU[i];
        }
    }
}

void Stylization::localStep()
{
    // Update R Matrix
    this->updateR();
    // Update z Vector
    this->updateZ();
    // Create Temp u Vector
    vector<Vector3d> tempU;
    this->updateU(tempU);
    // Update Penalty Rho and Scaled Dual u Vector
    this->updateRho(tempU);
}

void Stylization::globalStep()
{
    VectorXd rhs = VectorXd::Zero(3 * this->_deformedV.size());
    // Update the buffered deformedV data.
    function<Vector3d(const vector<Vector3d>&)> acc = [](const vector<Vector3d> &ls) -> Vector3d {
        Vector3d b = Vector3d::Zero();
        for (size_t i = 0; i < ls.size(); ++i) {
            b += ls[i];
        }
        return b;
    };
    for (size_t i = 0; i < this->_deformedV.size(); ++i) {
        // Build RHS.
        int k = 0;
        function<Vector3d(HalfEdge*)> iter = [this, &i, &k](HalfEdge* e) -> Vector3d {
            // Try to find endpoint vertex index.
            auto it = this->_index.find(e->next->vertex);
            const size_t j = it->second;
            // Update the index in the weight matrix.
            const Vector3d ret = this->_W[i](k, k) / 2. * (this->_R[i] + this->_R[j]) * (this->_V[j] - this->_V[i]);
            ++k;
            return ret;
        };
        const Vector3d b = this->_mesh->traverseFan(this->_mesh->getVertex(i), iter, acc);
        // Populate RHS.
        rhs.segment(3 * i, 3) << b;
    }
    // Solve the system.
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(this->_sysMatrix);
    MatrixXd temp(this->_sysMatrix);
    //VectorXd tempSum = temp.diagonal();
    VectorXd res = solver.solve(rhs);
    // Update deformedV.
    for (size_t i = 0; i < this->_deformedV.size(); ++i) {
        this->_deformedV[i] = res.segment(3 * i, 3);
    }
    // Update the mesh.
    this->updateDeformedV();
}

void Stylization::validate()
{
    assert(this->_mesh != NULL);
    cout << "Internal Lambda: " << this->_lambda << endl;
    const size_t n = this->_mesh->getNumVertices();
    assert(n != 0);
    assert(this->_rho.size() == n);
    assert(this->_V.size() == n);
    assert(this->_D.size() == n);
    assert(this->_n.size() == n);
    assert(this->_W.size() == n);
    assert(this->_R.size() == n);
    assert(this->_z.size() == n);
    assert(this->_u.size() == n);
    assert(this->_a.size() == n);
    assert(this->_deformedV.size() == n);
    assert(this->_deformedD.size() == n);
    assert(this->_deltaZ.size() == n);
}
