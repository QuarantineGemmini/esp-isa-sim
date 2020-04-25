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

struct matrix_cfg 
{
  reg_t base_addr;
  bool addr_valid;
  
  // Addressing modes
  enum AddrMode {ROW_MAJOR, IM2COL};
  AddrMode mode;

  // im2col-mode params
  uint32_t rows;
  uint32_t cols;
  uint16_t batch_size;
  uint16_t kernel_size;
  uint8_t channels;
  uint8_t padding;
  uint8_t stride;
  bool cfg_valid;
  
  // Methods
  void reset();
};

struct gemmini2_state_t
{
  enum GemminiState {LISTENING, COMPUTING, ERROR};
  GemminiState state;

  enum Dataflow {OS, WS};
  enum Activation {NONE, RELU, RELU6};
  void reset();

  // Matrice & Sizes
  matrix_cfg a, b, c, d;
  reg_t m, n, k;

  Dataflow mode;
  Activation act;
  reg_t acc_shift, sys_shift, relu6_shift;
  bool enable;

  // [ssteffl] TODO: HACK figure out better repeating_bias isa
  bool repeating_bias;
  
  // Config-valid flags
  bool size0_valid;
  bool size1_valid;
  bool config_ex_valid;
  bool bias_valid;
};

class gemmini2_t : public rocc_t
{
public:
  gemmini2_t() : cause(0), aux(0), debug(false) {}
  const char* name() { return "gemmini2"; }
  reg_t custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void reset();

  void setmode(reg_t rs1, reg_t rs2);
  void config_addr_mode(reg_t rs1, reg_t rs2);
  void compute(); 

private:
  gemmini2_state_t gemmini2_state;
  reg_t cause;
  reg_t aux;

  // Gemmini1 Opcodes
  const unsigned mvin_funct = 2;
  const unsigned mvout_funct = 3;
  const unsigned compute_preloaded_funct = 4;
  const unsigned compute_accumulated_funct = 5;
  const unsigned preload_funct = 6;
  const unsigned flush_funct = 7;

  // Shared Opcodes
  const unsigned setmode_funct = 0;
  
  // Gemmini2 Opcodes
  const unsigned config_addr_AB_funct = 10;
  const unsigned config_addr_CD_funct = 11;
  const unsigned config_size0_funct = 12;
  const unsigned config_size1_funct = 13;
  const unsigned config_repeating_bias_funct = 14;
  const unsigned config_reset = 15;
  const unsigned compute_funct = 16;
  const unsigned config_A_funct = 17;
  const unsigned config_B_funct = 18;
  const unsigned config_C_funct = 19;
  const unsigned config_D_funct = 20;

  bool debug;
  input_t apply_activation(input_t value);

  template <class T>
  T rounding_saturating_shift(accum_t value, uint64_t shift);

  template <class T>
  T read_from_dram(reg_t addr);

  template <class T>
  std::vector<std::vector<T>> *
  read_matrix_from_dram(matrix_cfg & mat, reg_t rows, reg_t cols, 
                        bool zeroable, bool repeating_bias);

  template <class T>
  void write_to_dram(reg_t addr, T data);
};

#endif
