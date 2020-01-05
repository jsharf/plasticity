#ifndef CODEGEN_CODEGEN_UTIL_H
#define CODEGEN_CODEGEN_UTIL_H
#include "codegen/codegen.h"

namespace codegen {

// (a <= index && index < b) ? then : ifnot.
std::string IfInRange(std::string index, std::string a, std::string b,
                      std::string then, std::string ifnot);

std::string BoundedArrayAccess(std::string array, std::string index,
                               std::string length);

// Row-order flattening.
std::string Flatten2d(std::string width, std::string height, std::string row,
                      std::string col);

// Assumes row-order flattening.
std::string Unflatten2dRow(std::string width, std::string height,
                           std::string i);

// Assumes row-order flattening.
std::string Unflatten2dCol(std::string width, std::string height,
                           std::string i);

}  // namespace codegen

#endif  // CODEGEN_CODEGEN_UTIL_H
