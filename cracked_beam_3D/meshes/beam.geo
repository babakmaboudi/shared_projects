// Gmsh project created on Mon Feb 12 13:30:38 2018
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0.1, 1.0};
//+
Point(4) = {0, 0, 0.1, 1.0};
//+
Point(5) = {0, 0.1, 0, 1.0};
//+
Point(6) = {1, 0.1, 0, 1.0};
//+
Point(7) = {0, 0.1, 0.1, 1.0};
//+
Point(8) = {1, 0.1, 0.1, 1.0};
Line(1) = {4, 7};
Line(2) = {7, 5};
Line(3) = {5, 1};
Line(4) = {1, 4};
Line(5) = {4, 3};
Line(6) = {3, 8};
Line(7) = {8, 6};
Line(8) = {6, 2};
Line(9) = {2, 3};
Line(10) = {1, 2};
Line(11) = {6, 5};
Line(12) = {7, 8};
Line Loop(13) = {4, 1, 2, 3};
Plane Surface(14) = {13};
Line Loop(15) = {5, 6, -12, -1};
Plane Surface(16) = {15};
Line Loop(17) = {4, 5, -9, -10};
Plane Surface(18) = {17};
Line Loop(19) = {9, 6, 7, 8};
Plane Surface(20) = {19};
Line Loop(21) = {3, 10, -8, 11};
Plane Surface(22) = {21};
Line Loop(23) = {7, 11, -2, 12};
Plane Surface(24) = {23};
Surface Loop(25) = {14, 18, 16, 20, 24, 22};
Volume(26) = {25};
Field[1] = Box;
Field[2] = LonLat;
Field[3] = MathEval;
Delete Field [2];
Delete Field [1];
Field[3].F = "x";
Field[3].F = "x^2";
Field[3].F = "-x^2";
Field[3].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-10*(z-0.05))))  )";
Background Field = 3;
Field[3].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-50*(z-0.05))))  )";
Field[3].F = "-x+1";
Field[3].F = "x^2+1";
Field[3].F = "x^3+1";
Field[3].F = "x^2+10";
Field[3].F = "x^2+100";
Field[3].F = "exp(x)";
Field[3].F = "exp(10*x)";
Field[3].F = "exp(100*x)";
Field[3].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-50*(z-0.05))))  )";
Field[3].F = "1.1 - (exp(-10*abs(x-0.5))   )";
Field[3].F = "1.1 - (exp(-10*abs(x))   )";
Field[3].F = "1.1 - (exp(10*abs(x))   )";
Field[3].F = "1.1 - (exp(10*abs(-x))   )";
Field[3].F = "1.1 - (exp(-10*abs(x))   )";
Field[3].F = "1.1 + (exp(-10*abs(x))   )";
Field[3].F = "1.1 + (exp(10*abs(x))   )";
Field[3].F = "1.1 - (exp(10*abs(x))   )";
Field[3].F = "1.1 - (exp(-10*abs(x))   )";
Field[3].F = "-1.1 - (exp(-10*abs(x))   )";
Field[3].F = "-1.1 +(exp(-10*abs(x))   )";
Field[3].F = "1.1 -(exp(-10*abs(x))   )";
Field[3].F = "-(exp(-10*abs(x))   )";
Field[3].F = "(exp(-10*abs(x))   )";
Field[3].F = "(exp(-abs(x))   )";
Field[3].F = "(exp(-0.1*abs(x))   )";
Field[3].F = "(exp(-0.5*abs(x))   )";
Field[1] = MathEval;
Field[1].F = "1/2*x+1";
Background Field = 1;
Background Field = -1;
Background Field = 1;
Field[1].F = "-1/2*x+1";
Field[1].F = "-1/4*x+1";
Field[1].F = "-1/3*x+1";
Field[1].F = "-1/2*x+1";
Field[1].F = "(exp(-abs(x))   )";
Field[1].F = "(exp(-10*abs(x))   )";
Field[1].F = "(exp(-2*abs(x))   )";
Field[1].F = "(exp(-abs(x))   )+1";
Field[1].F = "(exp(-abs(x+1))   )";
Field[1].F = "-1/2*x+1";
Field[1].F = "-1/2*(x+1)+1";
Field[1].F = "-1/2*(x+0.5)+1";
Field[1].F = "-1/2*(x)+1";
Field[1].F = "1";
Field[1].F = "0.1";
Field[1].F = "10";
Field[1].F = "100";
Field[1].F = "100000";
Field[1].F = "-1/2*x+1";
Field[1].F = "-1/2*(x+0.8)+1";
Field[1].F = "-1/2*(x+0.5)+1";
Field[1].F = "-1/2*(x+0.1)+1";
Field[1].F = "-1/2*(x+0.3)+1";
Field[1].F = "-1/2*(x+0.5)+1";
