#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include "codegen.h"

#include <string>

namespace codegen {

TEST_CASE("A cuda expression is generated.", "[codegen]") {
  CudaGenerator cgen;

  cgen.AppendLineOfCode(cgen.if_expr(cgen.equals("2", "0")));
  cgen.PushScope();
  cgen.PopScope();
  cgen.AppendLineOfCode(cgen.else_expr());
  cgen.PushScope();
  cgen.AppendLineOfCode(cgen.array_access_2d("W", "i", "j") + cgen.linesep());
  cgen.PopScope();

  std::string code = cgen.code();

  REQUIRE(code == "if(2==0)\n{}else\n{W[i][j];\n}");
}

}  // namespace codegen
