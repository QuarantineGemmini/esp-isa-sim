#ifndef _GEMMINI_H
#define _GEMMINI_H

#include "extension.h"
#include "rocc.h"
#include <random>
#include <limits>

// Systolic array input datatype (feeding into PEs, moving out of accumulator)
typedef int8_t input_t; 
// Systolic array output datatype (coming down from PEs, 
//  moving into accumulator)
typedef int16_t output_t; 
// Accumulator datatype (inside PEs for OS dataflow and for the 
//  external accumulator)
typedef int32_t accum_t; 

#ifdef RISCV_ENABLE_GEMMINI_COMMITLOG
#define dprintf(...) printf(__VA_ARGS__)
#else
#define dprintf(...)
#endif

struct gemmini2_state_t
{
  enum Dataflow {OS, WS};
  enum Activation {NONE, RELU, RELU6};
  void reset();

  // Matrix Addresses & Sizes
  reg_t a_addr, b_addr, c_addr, d_addr;
  reg_t m, n, k;

  Dataflow mode;
  Activation act;
  reg_t acc_shift, sys_shift, relu6_shift;
  bool enable;

  // [ssteffl] TODO: HACK figure out better repeating_bias isa
  bool repeating_bias;
};

class gemmini2_t : public rocc_t
{
public:
  gemmini2_t() : cause(0), aux(0), debug(false) {}
  const char* name() { return "gemmini2"; }
  reg_t custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void reset();

  void setmode(reg_t rs1, reg_t rs2);
  // TODO: use a different opcode than the gemmini_compute_preload...
  void compute(reg_t a_addr, reg_t bd_addr, bool preload);

private:
  gemmini2_state_t gemmini2_state;
  reg_t cause;
  reg_t aux;

  const unsigned setmode_funct = 0;
  const unsigned mvin_funct = 2;
  const unsigned mvout_funct = 3;
  const unsigned compute_preloaded_funct = 4;
  const unsigned compute_accumulated_funct = 5;
  const unsigned preload_funct = 6;
  const unsigned flush_funct = 7;

  const unsigned config_addr_AB_funct = 10;
  const unsigned config_addr_CD_funct = 11;
  const unsigned config_size0_funct = 12;
  const unsigned config_size1_funct = 13;
  const unsigned config_repeating_bias_funct = 14;
  const unsigned config_reset = 15;

  bool debug;
  input_t apply_activation(input_t value);

  template <class T>
  T rounding_saturating_shift(accum_t value, uint64_t shift);

  template <class T>
  T read_from_dram(reg_t addr);

  template <class T>
  std::vector<std::vector<T>> *
  read_matrix_from_dram(reg_t addr, reg_t rows, reg_t cols, 
                        bool zeroable, bool repeating_bias);

  template <class T>
  void write_to_dram(reg_t addr, T data);
};

#endif
