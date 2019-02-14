#ifndef CODEGEN_CODEGEN_H
#define CODEGEN_CODEGEN_H

#include <iostream>
#include <sstream>
#include <typeinfo>

namespace codegen {

class Generator {
 public:
  virtual ~Generator() {
    if (indent_level_ != 0) {
      std::cerr << "Uh oh, in ~Generator an object of type "
                << typeid(*this).name()
                << " has an unbalanced indent_level_ (!= 0). You must have "
                   "forgotten to call PopScope() or PushScope() somewhere."
                << std::endl;
      std::exit(1);
    }
  }

  // Used to control code scope. In Python these will affect whitespace
  // prepended to each line. In C++-like languages these will just add '{' and
  // '}'.
  virtual void PushScope() = 0;
  virtual void PopScope() = 0;

  // Used to append generic code with a new line at the end.
  // Whitespace-sensitive languages (python) will need to override this to add
  // appropriate indenting (based on current indent level set by PushScope()
  // and PopScope()).
  virtual void AppendLineOfCode(const std::string& code) {
    code_ << code << "\n";
  }

  // Used to append generic code with a new line at the end.
  // Whitespace-sensitive languages (python) will need to override this to add
  // appropriate indenting (based on current indent level set by PushScope()
  // and PopScope()).
  virtual void AppendCode(const std::string& code) { code_ << code; }

  // The following functions are all const and do not modify the state of the
  // object itself. Their functionality is described with the equivalent C++
  // syntax.
  // lhs = rhs;
  virtual std::string assign(const std::string& lhs,
                             const std::string& rhs) const = 0;
  // lhs += rhs;
  virtual std::string add_assign(const std::string& lhs,
                                 const std::string& rhs) const = 0;
  // value[index];
  virtual std::string array_access(const std::string& value,
                                   const std::string& index) const = 0;
  // value[index1][index2];
  virtual std::string array_access_2d(const std::string& value,
                                      const std::string& index1,
                                      const std::string& index2) const = 0;
  // if(term)
  virtual std::string if_expr(const std::string& term) const = 0;
  // (condition) ? a : b
  virtual std::string ternary(const std::string& condition,
                              const std::string& a,
                              const std::string& b) const = 0;
  // lhs && rhs; "and" is a reserved keyword so this was named op_and.
  virtual std::string op_and(const std::string& lhs,
                             const std::string& rhs) const = 0;
  // lhs || rhs; "or" is a reserved keyword so this was named op_or.
  virtual std::string op_or(const std::string& lhs,
                            const std::string& rhs) const = 0;
  // lhs == rhs;
  virtual std::string equals(const std::string& lhs,
                             const std::string& rhs) const = 0;
  // lhs < rhs;
  virtual std::string lt(const std::string& lhs,
                         const std::string& rhs) const = 0;
  // lhs > rhs;
  virtual std::string gt(const std::string& lhs,
                         const std::string& rhs) const = 0;
  // lhs <= rhs;
  virtual std::string lte(const std::string& lhs,
                          const std::string& rhs) const = 0;
  // lhs >= rhs;
  virtual std::string gte(const std::string& lhs,
                          const std::string& rhs) const = 0;
  // lhs + rhs;
  virtual std::string add(const std::string& lhs,
                          const std::string& rhs) const = 0;
  // lhs * rhs;
  virtual std::string mul(const std::string& lhs,
                          const std::string& rhs) const = 0;

  // lhs / rhs;
  virtual std::string div(const std::string& lhs,
                          const std::string& rhs) const = 0;

  virtual std::string mod(const std::string& lhs,
                          const std::string& rhs) const = 0;

  virtual std::string for_loop(const std::string &init,
                               const std::string &condition,
                               const std::string &next,
                               const std::string &body) const = 0;

  // "else"
  virtual std::string else_expr() const = 0;

  // ";"
  virtual std::string linesep() const = 0;

  std::string code() { return code_.str(); }

 protected:
  std::stringstream code_;
  int indent_level_ = 0;
};

class CudaGenerator : public Generator {
 public:
  // Control code scope.
  void PushScope() override;
  void PopScope() override;

  // The following functions are all const and do not modify the state of the
  // object itself. Their functionality is described with the equivalent C++
  // syntax.
  // lhs = rhs;
  std::string assign(const std::string& lhs,
                     const std::string& rhs) const override;
  // lhs += rhs;
  std::string add_assign(const std::string& lhs,
                         const std::string& rhs) const override;
  // value[index];
  std::string array_access(const std::string& value,
                           const std::string& index) const override;
  // value[index1][index2];
  std::string array_access_2d(const std::string& value,
                              const std::string& index1,
                              const std::string& index2) const override;
  // if(term)
  std::string if_expr(const std::string& term) const override;
  // (condition) ? a : b
  std::string ternary(const std::string& condition, const std::string& a,
                      const std::string& b) const override;
  // lhs && rhs; "and" is a reserved keyword so this was named op_and.
  std::string op_and(const std::string& lhs,
                     const std::string& rhs) const override;
  // lhs || rhs; "or" is a reserved keyword so this was named op_or.
  std::string op_or(const std::string& lhs,
                    const std::string& rhs) const override;
  // lhs == rhs;
  std::string equals(const std::string& lhs,
                     const std::string& rhs) const override;
  // lhs < rhs;
  std::string lt(const std::string& lhs, const std::string& rhs) const override;
  // lhs > rhs;
  std::string gt(const std::string& lhs, const std::string& rhs) const override;
  // lhs <= rhs;
  std::string lte(const std::string& lhs,
                  const std::string& rhs) const override;
  // lhs >= rhs;
  std::string gte(const std::string& lhs,
                  const std::string& rhs) const override;
  // lhs + rhs;
  std::string add(const std::string& lhs,
                  const std::string& rhs) const override;
  // lhs * rhs;
  std::string mul(const std::string& lhs,
                  const std::string& rhs) const override;

  // lhs / rhs;
  std::string div(const std::string& lhs,
                  const std::string& rhs) const override;

  // lhs % rhs;
  std::string mod(const std::string& lhs,
                  const std::string& rhs) const override;

  // "else"
  std::string else_expr() const override;

  std::string for_loop(const std::string &init, const std::string &condition,
                       const std::string &next,
                       const std::string &body) const override;

  // ";"
  std::string linesep() const override;
};

}  // namespace codegen

#endif  // CODEGEN_CODEGEN_H
