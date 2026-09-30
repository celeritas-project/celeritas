//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VgBasicTrackView.test.cc
//---------------------------------------------------------------------------//
#include "geocel/vg/VgBasicTrackView.hh"

#include "VecgeomTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class VgBasicTrackViewTest : public ::celeritas::test::Test
{
  protected:
    static std::string_view gdml_basename() { return "two-boxes"; }
};

TEST_F(VgBasicTrackViewTest, nothing) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
