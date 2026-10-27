//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.test.cc
//---------------------------------------------------------------------------//

#include "orange/surf/Tri.hh"

#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

class TriTest : public ::celeritas::test::Test
{
  protected:
    using Intersections = Tri<float>::Intersections;
    using Vertex = Tri<float>::Vertex;
};

TEST_F(TriTest, construction)
{
    Vertex a = {0.0, 0.0, 0.0};
    Vertex b = {1.0, 1.0, 1.0};
    Vertex c = {2.0, 2.0, 2.0};
    Tri<float> tri(a, b, c);
}

// TEST_F(TriTest, TEST_IF_CELER_DEVICE(device))
// {
//     TTestInput input;
//     t_test(input);
// }

//---------------------------------------------------------------------------//
}  // namespace celeritas
