#ifndef YOLOV1_H
#define YOLOV1_H

#include "nnet/nnet.h"

namespace nnet {
// This class implements the Yolov1 network architecture. Why not yolov4, in
// today's day and age? Because yolov4 is significantly more complex to
// implement and I don't feel like it.
// For more information, refer to the research paper:
//
// https://arxiv.org/pdf/1506.02640.pdf
//
// Inputs are scaled down to the input size of 448x448x3. It is expected that
// the input is serialized in 3 channels (red, green, then blue) with each
// channel containing a serialized bytestring of a two-dimensional image. The 2D
// image is serialized in rows (so starting from the top left to the top right,
// and then the second row on the left to the right and so on). This is
// implemented in plasticity/symbolic/symbolic_util.cc if you want to know
// exactly what I mean.

class Yolov1 {
 public:
  Yolov1();
  SetParams();
  Evaluate();
  TrainSample();

 private:
  Nnet net_;
};

}  // namespace nnet

#endif  /* YOLOV1_H */
