//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestMacros.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "testdetail/TestMacrosImpl.hh"  // IWYU pragma: export

using ::celeritas::testdetail::print_expected;

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//

//! Custom comparison for Celeritas test result types, using ADL
#define EXPECT_REF_EQ(EXPECTED, ACTUAL) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsRefEq, EXPECTED, ACTUAL)
#define EXPECT_REF_NEAR(EXPECTED, ACTUAL, TOL) \
    EXPECT_PRED_FORMAT3( \
        ::celeritas::testdetail::IsRefEq, EXPECTED, ACTUAL, TOL)

//! Container equality macro
#define EXPECT_VEC_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsVecEq, expected, actual)

//! Single-ULP floating point equality macro
#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
#    define EXPECT_REAL_EQ(expected, actual) EXPECT_DOUBLE_EQ(expected, actual)
#elif CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
#    define EXPECT_REAL_EQ(expected, actual) EXPECT_FLOAT_EQ(expected, actual)
#endif

//! Soft equivalence macro
#define EXPECT_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsSoftEquiv, expected, actual)

//! Soft equivalence macro with relative error or comparator
#define EXPECT_SOFT_NEAR(expected, actual, rel_error) \
    EXPECT_PRED_FORMAT3( \
        ::celeritas::testdetail::IsSoftEquiv, expected, actual, rel_error)

//! Container soft equivalence macro
#define EXPECT_VEC_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2( \
        ::celeritas::testdetail::IsVecSoftEquiv, expected, actual)

//! Container soft equivalence macro with relative error or comparator
#define EXPECT_VEC_NEAR(expected, actual, rel_error) \
    EXPECT_PRED_FORMAT3( \
        ::celeritas::testdetail::IsVecSoftEquiv, expected, actual, rel_error)

//! Helper for printing expected values in reference structs
#define CELER_ATTR_REPR(REF, ATTR) \
    ::celeritas::testdetail::StreamableAttr \
    { \
        #ATTR, REF.ATTR \
    }

//! Invoke an attribute comparison with an optional tolerance
#define CELER_INVOKE_ATTR_3(func, type, attr) func(#attr, &type::attr)
#define CELER_INVOKE_ATTR_4(func, type, attr, tol) \
    func(#attr, &type::attr, tol)
#define CELER_INVOKE_ATTR_IMPL(_1, _2, _3, _4, NAME, ...) NAME
#define CELER_INVOKE_ATTR(...) \
    CELER_INVOKE_ATTR_IMPL( \
        __VA_ARGS__, CELER_INVOKE_ATTR_4, CELER_INVOKE_ATTR_3, unused)( \
        __VA_ARGS__)

//! JSON string equality (soft equal for floats)
#define EXPECT_JSON_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsJsonEq, expected, actual)

//! Construct a test name that is disabled when assertions are enabled
#if CELERITAS_DEBUG
#    define TEST_IF_CELERITAS_DEBUG(name) name
#else
#    define TEST_IF_CELERITAS_DEBUG(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when CUDA/HIP are disabled
#if CELER_USE_DEVICE
#    define TEST_IF_CELER_DEVICE(name) name
#else
#    define TEST_IF_CELER_DEVICE(name) DISABLED_##name
#endif

//! Construct a test name that is disabled unless using double-precision real
#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
#    define TEST_IF_CELERITAS_DOUBLE(name) name
#else
#    define TEST_IF_CELERITAS_DOUBLE(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when Geant4 is disabled
#if CELERITAS_USE_GEANT4
#    define TEST_IF_CELERITAS_GEANT(name) name
#else
#    define TEST_IF_CELERITAS_GEANT(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when ROOT is disabled
#if CELERITAS_USE_ROOT
#    define TEST_IF_CELERITAS_USE_ROOT(name) name
#else
#    define TEST_IF_CELERITAS_USE_ROOT(name) DISABLED_##name
#endif
