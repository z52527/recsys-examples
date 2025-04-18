/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/
// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#ifdef HSTU_DISABLE_CONTEXT
  #define CONTEXT_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define CONTEXT_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_CAUSAL
  #define CAUSAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define CAUSAL_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_TARGET
  #define TARGET_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define TARGET_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_DELTA_Q
  #define DELTA_Q_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define DELTA_Q_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_RAB
  #define RAB_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define RAB_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_DRAB
  #define DRAB_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define DRAB_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_SM8x
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                    \
  [&] {                                                                        \
    constexpr static int ARCH_NAME = 90;                                       \
    return __VA_ARGS__();                                                      \
  }()
#else
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                    \
  [&] {                                                                        \
    if (ARCH == 86 || ARCH == 89) {                                            \
      constexpr static int ARCH_NAME = 86;                                     \
      return __VA_ARGS__();                                                    \
    } else if (ARCH < 90) {                                                    \
      constexpr static int ARCH_NAME = 80;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int ARCH_NAME = 90;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)                                           \
  [&] {                                                                        \
    if (HEADDIM == 64) {                                                       \
      constexpr static int kHeadSize = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 128) {                                               \
      constexpr static int kHeadSize = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 256) {                                               \
      constexpr static int kHeadSize = 256;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, NAME, ...)                              \
  [&] {                                                                        \
    bool useSeqLen = USE_VAR_SEQ_LEN;                                          \
    if (useSeqLen) {                                                           \
      using NAME = flash::VarSeqLenTraits;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME = flash::VarSeqLenTraits;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
