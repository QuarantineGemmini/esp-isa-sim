#include "gemmini2.h"
#include "mmu.h"
#include "trap.h"
#include <stdexcept>
#include <iostream>
#include <assert.h>

REGISTER_EXTENSION(gemmini2, []() {             \
  printf("Registering Gemmini2 ISA (im2col edition)\n\n");  \
  return new gemmini2_t;                        \
})

void matrix_cfg::reset()
{
  // Reset base-pointer and its validity 
  base_addr = 0;
  addr_valid = false;

  // Default addressing mode is row-major
  mode = ROW_MAJOR;

  // Scrub all im2col-addressing params 
  batch_size = 0;
  channels = 0;
  padding = 0;
  kernel_size = 0;
  stride = 0;
  
  // Allow "no-config" row-major setup
  cfg_valid = true;
}

void gemmini2_state_t::reset()
{
  enable = true;
  a.reset();
  b.reset();
  c.reset();
  d.reset();
  m = n = k = 0;

  size0_valid = false;
  size1_valid = false;
  config_ex_valid = false;
  bias_valid = false;
  
  // [ssteffl] TODO: remove OS dataflow in gemmini2. maybe support IS mode?
  mode = WS;
  act = NONE;
  acc_shift = 0;
  sys_shift = 0;
  relu6_shift = 0;
  repeating_bias = false;

  state = LISTENING;
  printf("Gemmini2 reset\n");
}

void gemmini2_t::reset() {
  gemmini2_state.reset();
}

template <class T>
T gemmini2_t::read_from_dram(reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load_uint8(addr + byte_idx) << (byte_idx*8);
  }
  return value;
}

template <class T>
std::vector<std::vector<T>> *
matrix_zeroes(reg_t rows, reg_t cols) {
  return new std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0));
}

template <class T>
std::vector<std::vector<T>> *
gemmini2_t::read_matrix_from_dram(matrix_cfg & mat, reg_t rows, reg_t cols, 
                                  bool zeroable, bool repeating_bias) {
  // Read and return Matrix of size `rows*cols` from address `addr` in main memory
  reg_t addr = mat.base_addr;
  
  // Initialize to all zeroes
  auto result = matrix_zeroes<T>(rows, cols);

  // if an input matrix is at addr 0, it is NULL, so don't do anything with 
  // it only the D matrix is zeroable; the A, B matrices must be valid
  if(addr == 0) {
    if(zeroable) {
      return result;
    }
    printf("ERROR: non-zeroable matrix given address zero!\n");
    exit(1);
  }

  if (mat.mode == matrix_cfg::AddrMode::ROW_MAJOR) {
    for (size_t i = 0; i < rows; i++) {
      auto ii = repeating_bias ? 0 : i;
      auto const dram_row_addr = addr + ii*sizeof(T)*cols;
      for (size_t j = 0; j < cols; j++) {
        auto const dram_byte_addr = dram_row_addr + j*sizeof(T);
        result->at(i).at(j) = gemmini2_t::read_from_dram<T>(dram_byte_addr);
      }
    }
  } else if (mat.mode == matrix_cfg::AddrMode::IM2COL) {

    // printf("SPIKE_GEMMINI2 IM2COL_PARAMS:    - [%u, %u, %u, %u, %u, %u]\n", 
    //     mat.batch_size, mat.padding, mat.rows, mat.kernel_size, mat.stride, mat.channels); // YAML-ish 
    int patch_row = 0;

    for (int n_batch = 0; n_batch < mat.batch_size; n_batch++) {
        for (int im_row = -mat.padding; im_row < mat.rows - mat.kernel_size + mat.padding + 1; im_row += mat.stride) {
            for (int im_col = -mat.padding; im_col < mat.cols - mat.kernel_size + mat.padding + 1; im_col += mat.stride) {
                int patch_col = 0;

                for (int im_channel = 0; im_channel < mat.channels; im_channel++) {
                    for (int filter_row = 0; filter_row < mat.kernel_size; filter_row++) {
                        for (int filter_col = 0; filter_col < mat.kernel_size; filter_col++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;
                            
                            if (pixel_row < 0 || pixel_row >= mat.rows
                                || pixel_col < 0 || pixel_col >= mat.cols) {

                            } else {
                                //output[patch_row][patch_col] = input[n_batch][pixel_row][pixel_col][im_channel];
                                auto const dram_byte_addr = mat.base_addr 
                                  + n_batch * mat.rows * mat.cols * mat.channels * sizeof(T) 
                                  + pixel_row * mat.cols * mat.channels * sizeof(T) 
                                  + pixel_col * mat.channels * sizeof(T) 
                                  + im_channel * sizeof(T);
                                result->at(patch_row).at(patch_col) = gemmini2_t::read_from_dram<T>(dram_byte_addr);
                            }           
                            patch_col++;
                        }
                    }
                }           
                patch_row++;
            }
        }
    }
  } else { 
    printf("ERROR: unsupported matrix addressing mode\n"); 
    exit(1);
  } 
  return result;
}

template <class T>
void gemmini2_t::write_to_dram(reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store_uint8(addr + byte_idx, (data >> (byte_idx*8)) & 0xFF);
  }
}

void gemmini2_t::setmode(reg_t rs1, reg_t rs2) {
  if ((rs1 & 0b11) == 0) { // rs1[1:0] == 2'b00, config_ex, configure execute pipeline
    gemmini2_state_t::Dataflow new_mode;
    gemmini2_state_t::Activation new_act;
    reg_t new_acc_shift, new_sys_shift, new_relu6_shift;

    // extract rs1[2], 0 = output stationary, 1 = weight stationary
    auto rs1_2 = (rs1 >> 2) & 0b1; 
    if (rs1_2 == 0) {
      //new_mode = gemmini2_state_t::OS;
      printf("GEMMINI: OS-mode not supported\n");
      illegal_instruction();
    } else {
      new_mode = gemmini2_state_t::WS;
    }

    // extract rs1[4:3], 0 = no activation, 1 = ReLU, 2 = ReLU6
    auto rs1_4_3 = (rs1 >> 3) & 0b11; 
    if (rs1_4_3 == 0) {
      new_act = gemmini2_state_t::NONE;
    } else if (rs1_4_3 == 1) {
      new_act = gemmini2_state_t::RELU;
    } else if (rs1_4_3 == 2) {
      new_act = gemmini2_state_t::RELU6;
    } else {
      assert(false);
    }

    new_acc_shift = (rs1 >> 32) & 0xFFFFFFFF;
    new_sys_shift = (rs2) & 0xFFFFFFFF;
    new_relu6_shift = (rs2 >> 32) & 0xFFFFFFFF;

    dprintf("GEMMINI: config_ex - set dataflow mode from %d to %d\n", 
        gemmini2_state.mode, new_mode);
    dprintf("GEMMINI: config_ex - set activation function from %d to %d\n", 
        gemmini2_state.act, new_act);
    dprintf("GEMMINI: config_ex - set acc_shift from %lu to %lu\n", 
        gemmini2_state.acc_shift, new_acc_shift);
    dprintf("GEMMINI: config_ex - set sys_shift from %lu to %lu\n", 
        gemmini2_state.sys_shift, new_sys_shift);
    dprintf("GEMMINI: config_ex - set relu6_shift from %lu to %lu\n", 
        gemmini2_state.relu6_shift, new_relu6_shift);

    gemmini2_state.mode = new_mode;
    gemmini2_state.act = new_act;

    assert(new_acc_shift >= 0 && new_acc_shift < sizeof(accum_t)*8);
    assert(new_sys_shift >= 0 && new_sys_shift < sizeof(output_t)*8);
    assert(new_relu6_shift >= 0);
    gemmini2_state.acc_shift = new_acc_shift;
    gemmini2_state.sys_shift = new_sys_shift;
    gemmini2_state.relu6_shift = new_relu6_shift;
    
    gemmini2_state.config_ex_valid = true;
  } 
  else if ((rs1 & 0b11) == 1) { 
    // rs1[1:0] == 2'b01, config_mvin, configure load pipeline
    printf("GEMMINI: config_mvin not supported!\n");
    illegal_instruction();
  } 
  else if ((rs1 & 0b11) == 2) { 
    // rs1[1:0] == 2'b10, config_mvout, configure store pipeline
    printf("GEMMINI: config_mvout not supported!\n");
    illegal_instruction();
  }
}

void gemmini2_t::compute() { 
  // `compute` performs Gemmini's core function - matrix multiply-add - 
  //  without referencing any underlying hardware detail.
  // 
  // * Operands A, B, and D are loaded from memory
  // * Multiply, add, activation, and any requested shifts are performed
  // * Result D is written back to memory
  // 
  // These computations are made independent of systolic array sizes, 
  // scratchpad-memory sizes, 
  // and any other microarchitectural detail (other than datatypes). 

  // FIXME: error check state has been set up
  
  // Load operands from memory
  auto A = read_matrix_from_dram<input_t>(gemmini2_state.a,
                                          gemmini2_state.m, 
                                          gemmini2_state.k, 
                                          false, false);
  auto B = read_matrix_from_dram<input_t>(gemmini2_state.b,
                                          gemmini2_state.k, 
                                          gemmini2_state.n, 
                                          false, false);
  auto D = read_matrix_from_dram<accum_t>(gemmini2_state.d,
                                          gemmini2_state.m, 
                                          gemmini2_state.n, 
                                          true, 
                                          gemmini2_state.repeating_bias);
  // Initialize an accumulator/ result 
  auto C = matrix_zeroes<input_t>(gemmini2_state.m, gemmini2_state.n);

  // Multiply & apply activation
  for (size_t i=0; i<gemmini2_state.m; i++) {
    for (size_t j=0; j<gemmini2_state.n; j++) {
      accum_t value = D->at(i).at(j);
      for (size_t k=0; k<gemmini2_state.k; k++) {
        value += ((accum_t)A->at(i).at(k)) * ((accum_t)B->at(k).at(j));
      }
      //input_t shifted = (gemmini2_state.mode == gemmini2_state_t::OS) ?
      //rounding_saturating_shift<input_t>(value, gemmini2_state.sys_shift):
      //rounding_saturating_shift<input_t>(value, 0);
      input_t shifted = rounding_saturating_shift<input_t>(value, 
                          gemmini2_state.acc_shift);
      input_t activated = apply_activation(shifted);
      C->at(i).at(j) = activated;
    }
  }
  
  // Write back to memory
  for (size_t i = 0; i < gemmini2_state.m; i++) {
    auto const dram_row_addr = gemmini2_state.c.base_addr + 
                               i*sizeof(input_t)*gemmini2_state.n;
    for (size_t j = 0; j < gemmini2_state.n; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(input_t);
      write_to_dram<input_t>(dram_byte_addr, C->at(i).at(j));
    }
  } 
}

void gemmini2_t::config_addr_mode(reg_t rs1, reg_t rs2) {
  // Set up addressing config, as dictated by CONFIG_ADDR RoCC commands

  // Extract which of our four matrices is being configured
  auto mat_num = (rs2 >> 60) & 0xF; 
  matrix_cfg * mat;
  if (mat_num==0) mat = &gemmini2_state.a;
  else if (mat_num==1) mat = &gemmini2_state.b;
  else if (mat_num==3) mat = &gemmini2_state.d;
  else if (mat_num==2) {
    // mat = &gemmini2_state.c;
    printf("GEMMINI ERROR: Storing non-row-major not supported, yet.");
    assert(false);
  } else {
    printf("GEMMINI ERROR: Attempting to configure invalid matrix");
    assert(false);
  }

  // Extract Addressing Mode
  auto mode = (rs2 >> 56) & 0xF; 
  if (mode != 1) { // For now, this better be setting IM2COL mode. Or else
    printf("CONFIG_ADDR GOT ILLEGAL MODE %u\n", mode);
    assert(false);
  }
  mat->mode = matrix_cfg::AddrMode::IM2COL;

  // FIXME: maybe do some bounds checks here, e.g. for zero sizes
  mat->rows = (rs1 >> 32);
  mat->cols = (rs1 & 0xFFFFFFFF);
  mat->stride = ((rs2 >> 48) & 0xFF);
  mat->padding = ((rs2 >> 40) & 0xFF);
  mat->channels = ((rs2 >> 32) & 0xFF);
  mat->kernel_size = ((rs2 >> 16) & 0xFFFF);
  mat->batch_size = ((rs2) & 0xFFFF);

}

reg_t gemmini2_t::custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  // Strip the dependency bits from the funct field 
  insn.funct = (insn.funct & 0b11111); 
  
  if (insn.funct == mvin_funct) {
    printf("GEMMINI: deprecated `mvin` instruction \n");
    illegal_instruction();
  }
  else if (insn.funct == mvout_funct) {
    printf("GEMMINI: deprecated `mvout` instruction \n");
    illegal_instruction();
  }
  else if (insn.funct == preload_funct) {
    printf("GEMMINI: deprecated `preload` instruction \n");
    illegal_instruction();
  }
  else if (insn.funct == flush_funct) {
    printf("GEMMINI: deprecated `flush` instruction. DO NOT USE\n");
  } 
  else if (insn.funct == compute_accumulated_funct) {
    printf("GEMMINI: deprecated `compute_acc` instruction\n");
    illegal_instruction();
  }
  else if (insn.funct == compute_preloaded_funct) {
    printf("GEMMINI: pending deprecation: `compute_preloaded` instruction\n");
    printf("GEMMINI: use `compute` instead\n");
    compute(); 
  }
  else if (insn.funct == compute_funct) {
    compute(); 
  }
  else if (insn.funct == setmode_funct) {
    setmode(xs1, xs2);
  }
  else if (insn.funct == config_addr_AB_funct) {
    gemmini2_state.a.base_addr = xs1;
    gemmini2_state.a.addr_valid = true; 
    gemmini2_state.b.base_addr = xs2;
    gemmini2_state.b.addr_valid = true; 
  } 
  else if (insn.funct == config_addr_CD_funct ){
    gemmini2_state.c.base_addr = xs1;
    gemmini2_state.c.addr_valid = true; 
    gemmini2_state.d.base_addr = xs2;
    gemmini2_state.d.addr_valid = true; 
  } 
  else if (insn.funct == config_size0_funct ){
    gemmini2_state.m = xs1;
    gemmini2_state.n = xs2;
    gemmini2_state.size0_valid = true;
  } 
  else if (insn.funct == config_size1_funct ){
    gemmini2_state.k = xs1;
    gemmini2_state.size1_valid = true;
  } 
  else if (insn.funct == config_repeating_bias_funct){
    gemmini2_state.repeating_bias = (bool)xs1;
    gemmini2_state.bias_valid = true;
  } 
  else if (insn.funct == config_A_funct){
    config_addr_mode(xs1, xs2);
  } 
  else if (insn.funct == config_reset) {
    reset();
  }
  else {
    printf("GEMMINI: encountered unknown instruction with funct: %d\n", insn.funct);
    illegal_instruction();
  }
  return 0;
}

// Applying activation from PE post-shifted output to scratchpad 
// (for OS dataflow)
// or from accumulator to DRAM (after shifting, for WS dataflow)
input_t gemmini2_t::apply_activation(input_t value) {
  if (gemmini2_state.act == gemmini2_state_t::RELU) {
    return value > 0 ? static_cast<input_t>(value) : static_cast<input_t>(0);
  } else if (gemmini2_state.act == gemmini2_state_t::RELU6) {
    auto positive = value > 0 ? value : static_cast<input_t>(0);
    return value > (6 << gemmini2_state.relu6_shift) ? 
      static_cast<input_t>(6 << gemmini2_state.relu6_shift) : 
      positive;
  } else if (gemmini2_state.act == gemmini2_state_t::NONE) {
    return static_cast<input_t>(value);
  } else assert(false);
}

template <class T>
T gemmini2_t::rounding_saturating_shift(accum_t value, uint64_t shift) {
  // Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
  int r = (shift == 0 ? 0 : ((value >> (shift-1)) & 1)) &
       (((shift <= 1 ? 0 : (value & ((1 << (shift-1)) - 1))) != 0) | ((value >> shift) & 1));
  accum_t shifted = (value >> shift) + r;

  // Saturate and cast element
  auto elem_t_max = std::numeric_limits<T>::max();
  auto elem_t_min = std::numeric_limits<T>::min();
  int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
  return static_cast<T>(elem);
}
