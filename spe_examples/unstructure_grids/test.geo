nx = 60;
ny = 60;

size_large = 10;
size_small = 2;

Point(1) = {0, 0, 0, size_large};
Point(2) = {nx, 0, 0, size_large};
Point(3) = {nx, ny, 0, size_large};
Point(4) = {0, ny, 0, size_large};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Point(5) = {nx / 2, ny / 2, 0, size_small};

Field[1] = Distance;
Field[1].NodesList = {5};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = size_small;
Field[2].LcMax = size_large;
Field[2].DistMin = ny / 10;
Field[2].DistMax = ny / 2;
Background Field = 2;

Mesh 2;
Save "adaptive_mesh.vtk";
