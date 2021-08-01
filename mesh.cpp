#include "mesh.h"

#include <iostream>
#include <fstream>

#include <unordered_set>
#include <unordered_map>

#include <QFileInfo>
#include <QString>

#define TINYOBJLOADER_IMPLEMENTATION
#include "util/tiny_obj_loader.h"
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

void Mesh::initFromVectors(const vector<Vector3d> &vertices,
           const vector<Vector3i> &faces)
{
    this->_vertices = vertices;
    this->_faces = faces;
}

void Mesh::loadFromFile(const string &filePath)
{
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    QFileInfo info(QString(filePath.c_str()));
    string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                                info.absoluteFilePath().toStdString().c_str(), (info.absolutePath().toStdString() + "/").c_str(), true);
    if (!err.empty()) {
        cerr << err << endl;
    }
    if (!ret) {
        cerr << "Failed to load/parse .obj file" << endl;
        return;
    }
    for (size_t s = 0; s < shapes.size(); ++s) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            unsigned int fv = shapes[s].mesh.num_face_vertices[f];
            Vector3i face;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset+v];
                face[v] = idx.vertex_index;
            }
            this->_faces.push_back(face);
            index_offset += fv;
        }
    }
    for (size_t i = 0; i < attrib.vertices.size(); i+=3) {
        this->_vertices.emplace_back(static_cast<double>(attrib.vertices[i]),
                                     static_cast<double>(attrib.vertices[i+1]),
                                     static_cast<double>(attrib.vertices[i+2]));
    }
    cout << "Loaded " << this->_faces.size() << " faces and " << this->_vertices.size() << " vertices" << endl;
}

void Mesh::saveToFile(const string &filePath)
{
    ofstream outfile;
    outfile.open(filePath);
    // Write Vertices
    for (size_t i = 0; i < this->_vertices.size(); ++i)
    {
        const Vector3d &v = this->_vertices[i];
        outfile << "v " << v[0] << " " << v[1] << " " << v[2] << endl;
    }
    // Write Faces
    for (size_t i = 0; i < this->_faces.size(); ++i)
    {
        const Vector3i &f = this->_faces[i];
        outfile << "f " << (f[0]+1) << " " << (f[1]+1) << " " << (f[2]+1) << endl;
    }
    outfile.close();
}

inline HalfEdge* createHalfEdge(Face* face, Vertex* vertex) {
    HalfEdge* edge = new HalfEdge();
    edge->face = face;
    edge->vertex = vertex;
    vertex->halfedge = edge;
    return edge;
}

// Helper Hash Function
inline long vertexPairIdHash(long countVertices, long i, long j) {
    return i * countVertices + j;
}

// Find a twin vector of given hash code of vertex pair based on given hash map.
inline HalfEdge* findTwin(unordered_map<long, HalfEdge*> &map, long vertexPairIdHash) {
    auto iter = map.find(vertexPairIdHash);
    return iter == map.end() ? NULL : iter->second;
}

HalfEdgeMesh* Mesh::convertToHalfEdgeMesh()
{
    vector<Face*> faces;
    vector<Vertex*> vertices;
    vector<HalfEdge*> edges;
    // Maps from hash code of vertex pair to edge pointer.
    unordered_map<long, HalfEdge*> vertexPairHash2HalfEdge;
    for (Vector3d p: this->_vertices) {
        // Populate the vertices
        Vertex* v = new Vertex();
        v->xyz = p;
        v->halfedge = NULL;
        vertices.push_back(v);
    }
    long numVertices = this->_vertices.size();
    for (Vector3i indices: this->_faces) {
        // Create Face
        Face* face = new Face();
        faces.push_back(face);
        // Create Half Edges
        HalfEdge* e0 = createHalfEdge(face, vertices[indices[0]]);
        HalfEdge* e1 = createHalfEdge(face, vertices[indices[1]]);
        HalfEdge* e2 = createHalfEdge(face, vertices[indices[2]]);
        face->halfedge = e0;
        // Populate Nexts
        e0->next = e1;
        e1->next = e2;
        e2->next = e0;
        // Solve Twins
        e0->twin = findTwin(vertexPairHash2HalfEdge, vertexPairIdHash(numVertices, indices[1], indices[0]));
        e1->twin = findTwin(vertexPairHash2HalfEdge, vertexPairIdHash(numVertices, indices[2], indices[1]));
        e2->twin = findTwin(vertexPairHash2HalfEdge, vertexPairIdHash(numVertices, indices[0], indices[2]));
        if (e0->twin != NULL)
            e0->twin->twin = e0;
        if (e1->twin != NULL)
            e1->twin->twin = e1;
        if (e2->twin != NULL)
            e2->twin->twin = e2;
        // Populate Twin Map
        vertexPairHash2HalfEdge.insert(make_pair(vertexPairIdHash(numVertices, indices[0], indices[1]), e0));
        vertexPairHash2HalfEdge.insert(make_pair(vertexPairIdHash(numVertices, indices[1], indices[2]), e1));
        vertexPairHash2HalfEdge.insert(make_pair(vertexPairIdHash(numVertices, indices[2], indices[0]), e2));
        // Populate Edge Vector
        edges.push_back(e0);
        edges.push_back(e1);
        edges.push_back(e2);
    }
    // Remember to delete after using.
    return new HalfEdgeMesh(faces, vertices, edges);
}

int Mesh::getNumFaces()
{
    return this->_faces.size();
}

HalfEdgeMesh::HalfEdgeMesh(const vector<Face*> &f, const vector<Vertex*> &v, const vector<HalfEdge*> &e)
{
    this->_faces = f;
    this->_vertices = v;
    this->_halfedges = e;
}

HalfEdgeMesh::~HalfEdgeMesh()
{
    for (HalfEdge* e: this->_halfedges) {
        delete e;
    }
    for (Face* f: this->_faces) {
        delete f;
    }
    for (Vertex* v: this->_vertices) {
        delete v;
    }
}

inline int findIndexFromMap(unordered_map<Vertex*, int> &map, Vertex* v) {
    auto iter = map.find(v);
    return iter == map.end() ? -1 : iter->second;
}

bool HalfEdgeMesh::convertToMesh(Mesh* m)
{
    vector<Vector3d> vertices;
    vector<Vector3i> faces;
    unordered_map<Vertex*, int> vertex2Ind;
    // Create Vertices and Buffer Index Map
    for (Vertex* v: this->_vertices) {
        vertex2Ind.insert(make_pair(v, vertices.size()));
        vertices.push_back(v->xyz);
    }
    for (Face* f: this->_faces) {
        const HalfEdge* e = f->halfedge;
        int ind0 = findIndexFromMap(vertex2Ind, e->vertex);
        int ind1 = findIndexFromMap(vertex2Ind, e->next->vertex);
        int ind2 = findIndexFromMap(vertex2Ind, e->next->next->vertex);
        // Throw Exceptions
        if (ind0 == -1 || ind1 == -1 || ind2 == -1) {
            return false;
        }
        if (ind0 >= (int) vertices.size() || ind0 >= (int) vertices.size() || ind0 >= (int) vertices.size()) {
            return false;
        }
        // Create Faces
        faces.push_back(Vector3i(ind0, ind1, ind2));
    }
    m->initFromVectors(vertices, faces);
    return true;
}

void HalfEdgeMesh::validate()
{
    // This function only validates the topological structure.
    unordered_map<HalfEdge*, Face*> allEdge2Face;
    unordered_map<Face*, int> allFaces2EdgeCount;
    unordered_map<Vertex*, unordered_set<HalfEdge*>*> allVertices2Edges;
    // Check Halfedges
    for (HalfEdge* e: this->_halfedges) {
        // Check NULL
        assert(e != NULL);
        assert(e->face != NULL);
        assert(e->next != NULL);
        assert(e->twin != NULL);
        assert(e->vertex != NULL);
        // Check Next/Loop
        assert(e->next != e);
        assert(e->next != NULL);
        assert(e->next->next != e);
        assert(e->next->next != NULL);
        assert(e->next->next->next == e);
        // Check Twin
        assert(e != e->twin);
        assert(e->twin != NULL);
        assert(e->twin->twin == e);
        // Check Face
        assert(e->face->halfedge == e || e->face->halfedge->next == e || e->face->halfedge->next->next == e);
        // Check Twin-Next Relation
        assert(e->next != e->twin);
        assert(e->next->vertex == e->twin->vertex);
        // Populate Edge->Face Map
        allEdge2Face.insert(make_pair(e, e->face));
        // Populate Vertex->Edges Map
        auto iterSet = allVertices2Edges.find(e->vertex);
        if (iterSet == allVertices2Edges.end()) {
            unordered_set<HalfEdge*>* temp = new unordered_set<HalfEdge*>();
            allVertices2Edges.insert(make_pair(e->vertex, temp));
            temp->insert(e);
        } else {
            iterSet->second->insert(e);
        }
        // Check Face-Edge Count
        auto iter = allFaces2EdgeCount.find(e->face);
        if (iter == allFaces2EdgeCount.end()) {
            allFaces2EdgeCount.insert(make_pair(e->face, 1));
        } else {
            ++iter->second;
        }
    }
    assert(this->_faces.size() == allFaces2EdgeCount.size());
    // Check faces
    for (Face* f: this->_faces) {
        // Check NULL
        assert(f != NULL);
        HalfEdge* e = f->halfedge;
        assert(e != NULL);
        // Check if Edge Checked
        auto iter0 = allEdge2Face.find(e);
        assert(iter0 != allEdge2Face.end());
        assert(iter0->second == f);
        // Check if Face All Stored
        auto iter1 = allFaces2EdgeCount.find(f);
        assert(iter1 != allFaces2EdgeCount.end());
        // Check Right Number of Edges Corresponding to Face
        assert(iter1->second == 3);
        // Check Loop All in Face
        assert(e->next->face == f);
        assert(e->next->next->face == f);
        // Check Twin Face
        assert(e->twin->face != e->face);
    }
    // Check vertices
    assert(allVertices2Edges.size() == this->_vertices.size());
    function<HalfEdge*(HalfEdge*)> helper = [&allVertices2Edges](HalfEdge* e) -> HalfEdge* {
        allVertices2Edges.find(e->vertex)->second->erase(e);
        return e;
    };
    for (Vertex* v: this->_vertices) {
        // Check NULL
        assert(v != NULL);
        HalfEdge* e = v->halfedge;
        assert(e != NULL);
        // Check if Vertex<->Edge and Edge Checked
        assert(allEdge2Face.find(e) != allEdge2Face.end());
        assert(e->vertex == v);
        // Check Triangle Vertex
        assert(e->vertex != e->next->vertex);
        assert(e->vertex != e->next->next->vertex);
        // Check Fan Property
        auto iter = allVertices2Edges.find(v);
        assert(iter != allVertices2Edges.end());
        function<bool(const vector<HalfEdge*> &)> aggregator = [e](const vector<HalfEdge*> &edges) -> bool {
            assert(edges.back()->twin->next == e);
            return true;
        };
        this->traverseFan(v, helper, aggregator);
        // Check Fin Property
        assert(allVertices2Edges.find(e->vertex)->second->empty());
    }
    for (auto p: allVertices2Edges) {
        delete p.second;
    }
}

int HalfEdgeMesh::getNumVertices()
{
    return this->_vertices.size();
}

Vertex* HalfEdgeMesh::getVertex(size_t i)
{
    return this->_vertices[i];
}

void HalfEdgeMesh::updateVertices(const std::vector<Eigen::Vector3d> &xyz)
{
    for (size_t i = 0; i < xyz.size(); ++i) {
        this->_vertices[i]->xyz = xyz[i];
    }
}

inline Vector3d getNormal(const Face* face) {
    const HalfEdge* e = face->halfedge;
    const Vector3d u = e->next->vertex->xyz - e->vertex->xyz;
    const Vector3d v = e->next->next->vertex->xyz - e->vertex->xyz;
    const Vector3d n = u.cross(v);
    return n / n.norm();
}

inline Matrix4d getQ(const Vector3d v, const Vector3d n) {
    Vector4d n4(n[0], n[1], n[2], -v.dot(n));
    return n4 * n4.transpose();
}

inline Vector4d findBestNewVertex(const Vector3d v1, const Vector3d v2, const Matrix4d Q) {
    Matrix4d D = Q;
    D(3,0) = 0.;
    D(3,1) = 0.;
    D(3,2) = 0.;
    D(3,3) = 1.;
    // Test if solvable
    if (D.determinant() < 1e-8) {
        Vector3d v = (v1 + v2) / 2.;
        return Vector4d(v[0], v[1], v[2], 1.);
    }
    // Solve the equation
    return D.inverse() * Vector4d(0., 0., 0., 1.);
}

inline void removeFromPQ(HalfEdge* e, unordered_map<HalfEdge*, multimap<double, HalfEdge*>::iterator> &lut, multimap<double, HalfEdge*> &pq) {
    auto it = lut.find(e);
    if (it == lut.end())
        return;
    pq.erase(it->second);
    lut.erase(it);
}

bool HalfEdgeMesh::testCollapse(const Vector3d pt, HalfEdge* e) {
    function<bool(HalfEdge*)> iterDeg = [](HalfEdge*) -> bool {
        return true;
    };
    function<int(const vector<bool>&)> aggDeg = [](const vector<bool> &ls) -> int {
        return ls.size();
    };
    // Check if endpoints have common neighbor with degree 3
    if (this->traverseFan(e->next->next->vertex, iterDeg, aggDeg) == 3) {
        return false;
    }
    // Check the other face
    if (this->traverseFan(e->twin->next->next->vertex, iterDeg, aggDeg) == 3) {
        return false;
    }
    // Check only two common neighbors
    unordered_set<Vertex*> b;
    function<bool(HalfEdge*)> iterInsertB = [&b](HalfEdge* e) -> bool {
        b.insert(e->twin->vertex);
        return true;
    };
    this->traverseFan(e->vertex, iterInsertB, aggDeg);
    int count = 0;
    function<bool(HalfEdge*)> iterCount = [&count, &b](HalfEdge* e) -> bool {
        if (b.find(e->twin->vertex) != b.end()) {
            ++count;
        }
        return true;
    };
    this->traverseFan(e->twin->vertex, iterCount, aggDeg);
    if (count != 2) {
        return false;
    }
    // Check potential geometric problems
    const Face* f1 = e->face;
    const Face* f2 = e->twin->face;
    // Handle faces around first vertex
    HalfEdge* current = e;
    do {
        if (current->face != f1 && current->face != f2) {
            // Check the normal condition
            Vector3d n = getNormal(current->face);
            Vector3d np = current->next->vertex->xyz - pt;
            np = np.cross(current->next->next->vertex->xyz - pt);
            if (n.dot(np) < 0.) {
                return false;
            }
        }
        current = current->twin->next;
    } while (current != e);
    // Handle faces around the other
    current = current->twin;
    do {
        if (current->face != f1 && current->face != f2) {
            // Check the normal condition
            Vector3d n = getNormal(current->face);
            Vector3d np = current->next->vertex->xyz - pt;
            np = np.cross(current->next->next->vertex->xyz - pt);
            if (n.dot(np) < 0.) {
                return false;
            }
        }
    } while (current != e->twin);
    return true;
}

void HalfEdgeMesh::collapse(HalfEdge* e) {
    HalfEdge* t = e->twin;
    // Performs collapse operation O(1) (sequential)
    Vertex* v1 = e->vertex;
    Vertex* v2 = e->twin->vertex;
    // Handle the two faces near the edge
    e->face->halfedge = NULL;
    t->face->halfedge = NULL;
    // Merge edges of two faces
    e->next->twin->twin = e->next->next->twin;
    e->next->next->twin->twin = e->next->twin;
    t->next->twin->twin = t->next->next->twin;
    t->next->next->twin->twin = t->next->twin;
    // Resolve vertex issues
    if (e->next->next->vertex->halfedge == e->next->next) {
        e->next->next->vertex->halfedge = e->next->next->twin->twin;
    }
    if (t->next->next->vertex->halfedge == t->next->next) {
        t->next->next->vertex->halfedge = t->next->next->twin->twin;
    }
    if (v1->halfedge == e || v1->halfedge == t->next) {
        v1->halfedge = e->next->twin->twin;
    }
    // Handle other edges' vertex issue
    for (HalfEdge* temp = t->next->twin->twin; temp->vertex == v2; temp = temp->next->next->twin) {
        temp->vertex = v1;
    }
    // Labels corresponding halfedges and vertex (sequential roughly O(1))
    v2->halfedge = NULL;
    while (e->twin != NULL) {
        e->twin = NULL;
        t->twin = NULL;
        e = e->next;
        t = t->next;
    }
}

void HalfEdgeMesh::clean() {
    // Remove faces from the buffer
    vector<Face*> newFaces;
    for (size_t i = 0; i < this->_faces.size(); ++i) {
        if (this->_faces[i]->halfedge == NULL) {
            delete this->_faces[i];
            continue;
        }
        newFaces.push_back(this->_faces[i]);
    }
    this->_faces = newFaces;
    // Remove vertices
    vector<Vertex*> newVertices;
    for (size_t i = 0; i < this->_vertices.size(); ++i) {
        if (this->_vertices[i]->halfedge == NULL) {
            delete this->_vertices[i];
            continue;
        }
        newVertices.push_back(this->_vertices[i]);
    }
    this->_vertices = newVertices;
    // Remove edges
    vector<HalfEdge*> newEdges;
    for (size_t i = 0; i < this->_halfedges.size(); ++i) {
        if (this->_halfedges[i]->twin == NULL) {
            delete this->_halfedges[i];
            continue;
        }
        newEdges.push_back(this->_halfedges[i]);
    }
    this->_halfedges = newEdges;
}

void HalfEdgeMesh::simplify(int n)
{
    // Pre-calculate all Q matrices for each face in original mesh O(n)
    unordered_map<Face*, Matrix4d> faceQ;
    for (size_t i = 0; i < this->_faces.size(); ++i) {
        Face* f = this->_faces[i];
        const Vector3d v = f->halfedge->vertex->xyz;
        const Vector3d n = getNormal(f);
        faceQ.insert(make_pair(f, getQ(v, n)));
    }
    // Pre-calculate Vertex Q O(n)
    unordered_map<Vertex*, Matrix4d> vertexQ;
    for (size_t i = 0; i < this->_vertices.size(); ++i) {
        // Accumulate vertex Q from nearby face Q matrices
        function<Matrix4d(HalfEdge*)> iter = [&faceQ](HalfEdge* e) -> Matrix4d {
            return faceQ.find(e->face)->second;
        };
        function<Matrix4d(const vector<Matrix4d>&)> agg = [](const vector<Matrix4d> &ls) -> Matrix4d {
            Matrix4d Q = Matrix4d::Zero();
            for (Matrix4d m: ls) {
                Q += m;
            }
            return Q;
        };
        Matrix4d Q = this->traverseFan(this->_vertices[i], iter, agg);
        vertexQ.insert(make_pair(this->_vertices[i], Q));
    }
    // Initialize the priority queue (pseudo actually an ordered multiset/multimap RBT) O(nlogn)
    multimap<double, HalfEdge*> pq;
    unordered_map<HalfEdge*, multimap<double, HalfEdge*>::iterator> lut;
    unordered_set<HalfEdge*> flag;
    unordered_map<HalfEdge*, Vector3d> edge2newPoint;
    for (size_t i = 0; i < this->_halfedges.size(); ++i) {
        HalfEdge* e = this->_halfedges[i];
        // Avoid double handling
        if (flag.find(e) != flag.end()) {
            continue;
        }
        flag.insert(e);
        flag.insert(e->twin);
        // Calculate best midpoint and corresponding error score
        const Matrix4d Q = vertexQ.find(e->vertex)->second + vertexQ.find(e->twin->vertex)->second;
        Vector4d newPoint = findBestNewVertex(e->vertex->xyz, e->twin->vertex->xyz, Q);
        edge2newPoint.insert(make_pair(e, newPoint.head<3>()));
        double err = newPoint.transpose() * Q * newPoint;
        lut.insert(make_pair(e, pq.insert(make_pair(err, e))));
    }
    // Start removing
    int numFaces = n;
    for (int numRemoved = 0; numRemoved < numFaces; ++numRemoved) {
        // Select the edge with lowest error
        if (pq.empty())
            break;
        // Find the next candidate (roughly de facto O(1) assuming normal mesh)
        auto iter = pq.begin();
        HalfEdge* e;
        while (iter != pq.end()) {
            e = iter->second;
            Vector3d pt = edge2newPoint.find(e)->second;
            if (this->testCollapse(pt, e)) {
                break;
            }
            ++iter;
        }
        // Could not find anything good
        if (iter == pq.end())
            break;
        HalfEdge* t = e->twin;
        // Performs collapse operation O(1) (sequential)
        Vertex* v1 = e->vertex;
        v1->xyz = edge2newPoint.find(e)->second;
        this->collapse(e);
        // Remove hidden edges in the queue O(logn)
        removeFromPQ(e, lut, pq);
        removeFromPQ(t, lut, pq);
        removeFromPQ(e->next, lut, pq);
        removeFromPQ(e->next->next, lut, pq);
        removeFromPQ(t->next, lut, pq);
        removeFromPQ(t->next->next, lut, pq);
        // Update Q for each related face (roughly O(1) assuming normal mesh)
        e = v1->halfedge;
        HalfEdge* current = e;
        do {
            Face* f = current->face;
            const Vector3d v = f->halfedge->vertex->xyz;
            const Vector3d n = getNormal(f);
            faceQ[f] = getQ(v, n);
        } while (e != current);
        // Update Q for new vertex (roughly O(1) assuming normal mesh)
        Matrix4d Q = Matrix4d::Zero();
        e = v1->halfedge;
        current = e;
        do {
            Q += faceQ.find(current->face)->second;
            current = current->twin->next;
        } while (e != current);
        vertexQ[v1] = Q;
        // Update Q for other related vertices (roughly O(1) assuming normal mesh)
        e = e->twin;
        current = e;
        do {
            Matrix4d Q = Matrix4d::Zero();
            HalfEdge* ve = current;
            do {
                Q += faceQ.find(ve->face)->second;
                ve = ve->twin->next;
            } while (ve != current);
            vertexQ[current->vertex] = Q;
            current = current->next->twin;
        } while (e != current);
        // Update error scores for each spoke edge
        do {
            // Remove original data in the heap O(logn) assuming normal mesh
            removeFromPQ(current, lut, pq);
            removeFromPQ(current->twin, lut, pq);
            // Calculate best midpoint and corresponding error score
            const Matrix4d Q = vertexQ.find(current->vertex)->second + vertexQ.find(current->twin->vertex)->second;
            Vector4d newPoint = findBestNewVertex(current->vertex->xyz, current->twin->vertex->xyz, Q);
            edge2newPoint[current] = newPoint.head<3>();
            double err = newPoint.transpose() * Q * newPoint;
            lut.insert(make_pair(current, pq.insert(make_pair(err, current))));
            current = current->next->twin;
        } while (e != current);
        // Update error scores for each outer edge
        e = e->twin->next;
        current = e;
        do {
            // Remove original data in the heap O(logn) assuming normal mesh
            removeFromPQ(current, lut, pq);
            removeFromPQ(current->twin, lut, pq);
            // Calculate best midpoint and corresponding error score
            const Matrix4d Q = vertexQ.find(current->vertex)->second + vertexQ.find(current->twin->vertex)->second;
            Vector4d newPoint = findBestNewVertex(current->vertex->xyz, current->twin->vertex->xyz, Q);
            edge2newPoint[current] = newPoint.head<3>();
            double err = newPoint.transpose() * Q * newPoint;
            lut.insert(make_pair(current, pq.insert(make_pair(err, current))));
            current = current->next->twin->next;
        } while (e != current);
    }
    this->clean();
}

inline float calculateWeight(const int n) {
    if (n == 3) {
        return .1875;
    }
    float temp = .375 + cos(2 * M_PI / n) * .25;
    return (.625 - temp * temp) / n;
}

void HalfEdgeMesh::subdivide(int iter) {
    if (iter == 0) {
        return;
    }
    // Store the old size of vertices buffer
    size_t oldSize = this->_vertices.size();
    // Buffer the new position of old vertices
    vector<Vector3d> buffer;
    for (size_t i = 0; i < oldSize; ++i) {
        Vertex* v = this->_vertices[i];
        HalfEdge* e = v->halfedge;
        // Calculate the degree of the vertex
        HalfEdge* current = e;
        vector<Vector3d> points;
        int n = 0;
        do {
            ++n;
            points.push_back(current->next->vertex->xyz);
            current = current->twin->next;
        } while (current != e);
        // Calculate weight and new position.
        float weight = calculateWeight(n);
        Vector3d newPos = Vector3d(0., 0., 0.);
        for (size_t j = 0; j < points.size(); ++j) {
            newPos += points[j];
        }
        buffer.push_back(v->xyz * (1. - n * weight) + newPos * weight);
    }
    // Create new vertices
    unordered_set<HalfEdge*> flag;
    size_t oldEdgeSize = this->_halfedges.size();
    for (size_t i = 0; i < oldEdgeSize; ++i) {
        HalfEdge* e = this->_halfedges[i];
        // Avoid double handling
        if (flag.find(e) != flag.end()) {
            continue;
        }
        flag.insert(e);
        flag.insert(e->twin);
        // Add new vertex and assign position
        this->_vertices.push_back(new Vertex());
        Vertex* newVertex = this->_vertices.back();
        newVertex->xyz = .375 * (e->vertex->xyz + e->twin->vertex->xyz) + .125 * (e->next->next->vertex->xyz + e->twin->next->next->vertex->xyz);
        // Add new edge and resolve loop issue
        this->_halfedges.push_back(new HalfEdge());
        HalfEdge* newEdge = this->_halfedges.back();
        newEdge->vertex = newVertex;
        newEdge->next = e->next;
        e->next = newEdge;
        newVertex->halfedge = newEdge;
        // Add new edge for twin side and resolve loop issue for twin side
        this->_halfedges.push_back(new HalfEdge);
        newEdge = this->_halfedges.back();
        newEdge->vertex = newVertex;
        newEdge->next = e->twin->next;
        e->twin->next = newEdge;
        // Assign twin
        e->next->twin = e->twin;
        newEdge->twin = e;
        e->twin->twin = e->next;
        e->twin = newEdge;
    }
    // Now each face should have 6 edges
    // Update the positions of old vertices
    for (size_t i = 0; i < oldSize; ++i) {
        this->_vertices[i]->xyz = buffer[i];
    }
    // Solve the faces through given stacked half-edges
    size_t numFaces = this->_faces.size();
    for (size_t i = 0; i < numFaces; ++i) {
        // Start from a midpoint edge
        HalfEdge* e = this->_faces[i]->halfedge->next;
        // Build inner triangle
        this->_faces.push_back(new Face());
        HalfEdge* newEdge[6];
        for (int j = 0; j < 3; ++j) {
            // Add new halfedge
            this->_halfedges.push_back(new HalfEdge());
            newEdge[j] = this->_halfedges.back();
            newEdge[j]->vertex = e->vertex;
            newEdge[j]->face = this->_faces.back();
            // Add new twin
            this->_halfedges.push_back(new HalfEdge());
            newEdge[3+j] = this->_halfedges.back();
            // Build outer loops
            newEdge[3+j]->next = e;
            // Go to next midpoint
            e = e->next->next;
            newEdge[3+j]->vertex = e->vertex;
            newEdge[j]->twin = newEdge[3+j];
            newEdge[3+j]->twin = newEdge[j];
        }
        this->_faces.back()->halfedge = newEdge[0];
        // Build inner loops
        newEdge[0]->next = newEdge[1];
        newEdge[1]->next = newEdge[2];
        newEdge[2]->next = newEdge[0];
        // Build other triangles
        newEdge[5]->face = this->_faces[i];
        this->_faces.push_back(new Face());
        newEdge[3]->face = this->_faces.back();
        this->_faces.back()->halfedge = newEdge[3];
        this->_faces.push_back(new Face());
        newEdge[4]->face = this->_faces.back();
        this->_faces.back()->halfedge = newEdge[4];
        // Resolve the straight highway issue in the outer triangles
        e = e->next;
        for (int j = 3; j < 6; ++j) {
            HalfEdge* temp = e->next->next;
            e->next = newEdge[j];
            e->face = newEdge[j]->face;
            newEdge[j]->next->face = e->face;
            e = temp;
        }
    }
    this->subdivide(iter - 1);
}
