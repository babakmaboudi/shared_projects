cl__1 = 1;
Point(3) = {0, 0, 0, 1};
Point(4) = {1, 0, 0, 1};
Point(5) = {1, 0.2, 0, 1};
Point(6) = {0, 0.2, 0, 1};
Point(7) = {0, 0.2, 0.2, 1};
Point(8) = {1, 0.2, 0.2, 1};
Point(9) = {1, 0, 0.2, 1};
Point(10) = {0, 0, 0.2, 1};
Point(11) = {0.48, 0, 0.2, 1};
Point(12) = {0.48, 0.2, 0.2, 1};
Point(13) = {0.51, 0.2, 0.2, 1};
Point(14) = {0.51, 0, 0.2, 1};
Point(15) = {0.5, 0, 0.1, 1};
Point(16) = {0.5, 0.2, 0.1, 1};
Line(1) = {5, 4};
Line(2) = {4, 9};
Line(3) = {9, 8};
Line(4) = {8, 5};
Line(5) = {5, 6};
Line(6) = {4, 3};
Line(7) = {3, 10};
Line(8) = {10, 7};
Line(9) = {7, 6};
Line(10) = {7, 12};
Line(11) = {12, 16};
Line(12) = {16, 13};
Line(13) = {13, 8};
Line(14) = {9, 14};
Line(15) = {14, 15};
Line(16) = {15, 11};
Line(17) = {11, 10};
Line(18) = {3, 6};
Line(19) = {16, 15};
Line(20) = {14, 13};
Line(21) = {12, 11};
Line Loop(23) = {14, 20, 13, -3};
Plane Surface(23) = {23};
Line Loop(25) = {19, -15, 20, -12};
Plane Surface(25) = {25};
Line Loop(27) = {11, 19, 16, -21};
Plane Surface(27) = {27};
Line Loop(29) = {21, 17, 8, 10};
Plane Surface(29) = {29};
Line Loop(31) = {7, -17, -16, -15, -14, -2, 6};
Plane Surface(31) = {31};
Line Loop(33) = {10, 11, 12, 13, 4, 5, -9};
Plane Surface(33) = {33};
Line Loop(35) = {1, 2, 3, 4};
Plane Surface(35) = {35};
Line Loop(37) = {8, 9, -18, 7};
Plane Surface(37) = {37};
Line Loop(39) = {5, -18, -6, -1};
Plane Surface(39) = {39};
Surface Loop(41) = {37, 39, 27, 29, 31, 33, 35, 23, 25};
Volume(41) = {41};
Field[1] = MathEval;
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-50*(z-0.01))))  )";
Background Field = 1;
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-100*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-10*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-1*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-100*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-1000*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-1*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-100*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "0.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.01))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.05))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.01*(z-0.1))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.1*(z-0.1))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.1)) *  (1/(1+exp(-0.1*(z-0.1))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.1*(z-0.1))))  )";
//+
Field[2] = Attractor;
//+
Delete Field [2];
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-0.1*(z-0.05))))  )";
//+
Field[1].F = "1.1 - (exp(-10*abs(x-0.5)) *  (1/(1+exp(-1*(z-0.05))))  )";
