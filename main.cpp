#include <QCoreApplication>
#include <QCommandLineParser>

#include <iostream>
#include <chrono>

#include "mesh.h"
#include "stylization.h"

#define DEFAULT_LAMBDA .2
#define DEFAULT_COARSE_NUM_FACES_RATE .8

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addPositionalArgument("infile", "Input .obj file path");
    parser.addPositionalArgument("outfile", "Output .obj file path");
    parser.addPositionalArgument("method", "regular/fast");

    // Lambda value
    parser.addPositionalArgument("args1", "Lambda value");

    // Fast Algorithm Only: Target Number of Faces
    parser.addPositionalArgument("args2", "target number of faces for coarse process");

    parser.process(a);

    const QStringList args = parser.positionalArguments();
    if (args.size() < 3) {
        cerr << "Arguments <input .obj file path> <output .obj file path> <method (regular/fast)> <method-specific arguments...>" << endl;
        a.exit(1);
        return 1;
    }
    QString infile = args[0];
    QString outfile = args[1];
    QString method = args[2];

    Mesh m;
    m.loadFromFile(infile.toStdString());

    double lambda;
    bool ok = false;
    if (args.size() > 3) {
        lambda = args[3].toDouble(&ok);
    }
    if (!ok || lambda < 0.) {
        lambda = DEFAULT_LAMBDA;
    }
    int numFacesForFastPre;
    ok = false;
    if (args.size() > 4) {
        numFacesForFastPre = args[4].toDouble(&ok);
    }
    if (!ok || numFacesForFastPre < 5) {
        numFacesForFastPre = static_cast<int>(DEFAULT_COARSE_NUM_FACES_RATE * m.getNumFaces());
    }

    // Convert the mesh into your own data structure.
    HalfEdgeMesh* mesh = m.convertToHalfEdgeMesh();
    mesh->validate();

    auto t0 = high_resolution_clock::now();

    // Implement the operations.
    Stylization style(mesh);
    cout << "Lambda: " << lambda << endl;
    if (method == "regular") {
        bool omitPrint = args.size() < 5 || args[4].toInt() == 0;
        style.transform(lambda, omitPrint);
    } else if (method == "fast") {
        cout << "m: " << numFacesForFastPre << endl;
        bool omitPrint = args.size() < 6 || args[5].toInt() == 0;
        style.fastTransform(lambda, numFacesForFastPre, omitPrint);
    } else {
        cerr << "Error: Unknown method \"" << method.toUtf8().constData() << "\"" << endl;
    }

    auto t1 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t1 - t0).count();
    cout << "Execution took " << duration << " milliseconds." << endl;

    // Convert your data structure back to the basic format.
    mesh->validate();
    if (!mesh->convertToMesh(&m)) {
        cerr << "Error: Unknown problem while converting result half-edge mesh back to default mesh!" << endl;
    }
    delete mesh;
    m.saveToFile(outfile.toStdString());

    a.exit();
}
