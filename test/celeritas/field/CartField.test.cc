//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartField.test.cc
//---------------------------------------------------------------------------//

#include <cmath>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

#include "Test.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

#if CELERITAS_USE_COVFIE
using CartMapFieldTest = ::celeritas::test::Test;
#    define CartMapFieldTest CartMapFieldTest_
#else
using DISABLED_CartMapFieldTest = ::celeritas::test::Test;
using CartMapFieldTest_ = ::celeritas::test::Test;
#    define CartMapFieldTest DISABLED_CartMapFieldTest

TEST_F(CartMapFieldTest_, all)
{
    // need at least one test for gtest to succeed
    SUCCEED();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
