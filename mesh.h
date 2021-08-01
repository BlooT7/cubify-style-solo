#ifndef MESH_H
#define MESH_H

#include <string>
#include <iostream>

#include <vector>
#include <Eigen/StdVector>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3i)

struct HalfEdge;

typedef struct Vertex {
    HalfEdge *halfedge;
    Eigen::Vector3d xyz;
} Vertex;

typedef struct Face {
    HalfEdge* halfedge;
} Face;

typedef struct HalfEdge {
    HalfEdge *twin;
    HalfEdge *next;
    Vertex *vertex;
    Face *face;
} HalfEdge;

class Mesh;

class HalfEdgeMesh
{
public:
    HalfEdgeMesh(const std::vector<Face*> &f, const std::vector<Vertex*> &v, const std::vector<HalfEdge*> &e);
    ~HalfEdgeMesh();
    bool convertToMesh(Mesh* m);
    // A helper function to traverse the fan based on given vertex: we also need what to do with each edge surronded by the vertex, and the results would
    // be stored in a vector container with type P. Then, another function will handle the vector which stores all outputs from the previous function.
    // It is defined here for convenience of compiling.
    template<typename T, typename P, typename A>
        T traverseFan(Vertex* v, std::function<P(HalfEdge*)> func, std::function<T(const std::vector<P, A> &)> aggregator) {
            HalfEdge* current = v->halfedge;
            const HalfEdge* start = current;
            std::vector<P> results;
            do {
                results.push_back(func(current));
                current = current->twin->next;
            } while (current != NULL && current != start);
            return aggregator(results);
    }
    template<typename T, typename P, typename A>
        T traverseFanAllEdge(Vertex* v, std::function<P(HalfEdge*)> func, std::function<T(const std::vector<P, A> &)> aggregator) {
            HalfEdge* current = v->halfedge;
            const HalfEdge* start = current;
            std::vector<P> results;
            do {
                results.push_back(func(current));
                current = current->next;
                results.push_back(func(current));
                current = current->next;
                results.push_back(func(current));
                current = current->twin;
            } while (current != NULL && current != start);
            return aggregator(results);
    }
    void validate();
    int getNumVertices();
    Vertex* getVertex(size_t i);
    void updateVertices(const std::vector<Eigen::Vector3d> &xyz);
    void simplify(int n);
    bool testCollapse(const Eigen::Vector3d pt, HalfEdge* e);
    void collapse(HalfEdge* e);
    void clean();
    void subdivide(int iter);
private:
    std::vector<HalfEdge*> _halfedges;
    std::vector<Vertex*> _vertices;
    std::vector<Face*> _faces;
};

class Mesh
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void initFromVectors(const std::vector<Eigen::Vector3d> &vertices,
         const std::vector<Eigen::Vector3i> &faces);
    void loadFromFile(const std::string &filePath);
    void saveToFile(const std::string &filePath);
    HalfEdgeMesh* convertToHalfEdgeMesh();
    int getNumFaces();
private:
    std::vector<Eigen::Vector3d> _vertices;
    std::vector<Eigen::Vector3i> _faces;
};

#endif // MESH_H
