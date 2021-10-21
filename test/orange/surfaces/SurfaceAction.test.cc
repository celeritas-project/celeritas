//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceAction.test.cc
//---------------------------------------------------------------------------//
#include "orange/surfaces/SurfaceAction.hh"

#include <sstream>
#include <string>
#include <vector>

#include "base/CollectionMirror.hh"
#include "base/Range.hh"
#include "orange/Data.hh"
#include "orange/construct/SurfaceInserter.hh"
#include "orange/surfaces/Surfaces.hh"
#include "orange/surfaces/SurfaceIO.hh"
#include "celeritas_test.hh"
// #include "SurfaceAction.test.hh"

using namespace celeritas;
// using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceActionTest : public celeritas::Test
{
  protected:
    using SurfaceDataMirror = CollectionMirror<SurfaceData>;

    void SetUp() override
    {
        SurfaceData<Ownership::value, MemSpace::host> surface_data;
        SurfaceInserter                               insert(&surface_data);
        insert(PlaneX(1));
        insert(PlaneY(2));
        insert(PlaneZ(3));
        insert(CCylX(5));
        insert(CCylY(6));
        insert(CCylZ(7));
        insert(GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9));

        surf_params_ = SurfaceDataMirror{std::move(surface_data)};
    }

    SurfaceDataMirror surf_params_;
};

//---------------------------------------------------------------------------//
// HELPERS
//---------------------------------------------------------------------------//

struct ToString
{
    template<class S>
    std::string operator()(S&& surf) const
    {
        std::ostringstream os;
        os << surf;
        return os.str();
    }

    // Make sure this test class is being move-constructed
    ToString()                = default;
    ToString(const ToString&) = delete;
    ToString(ToString&&)      = default;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceActionTest, string)
{
    // Create functor
    Surfaces surfaces(surf_params_.host());
    auto     surf_to_string = make_surface_action(surfaces, ToString{});

    // Loop over all surfaces and apply
    std::vector<std::string> strings;
    for (auto id : range(SurfaceId{surfaces.num_surfaces()}))
    {
        strings.push_back(surf_to_string(id));
    }

    // clang-format off
    const std::string expected_strings[] = {
        "Plane: x=1",
        "Plane: y=2",
        "Plane: z=3",
        "Cyl x: r=5",
        "Cyl y: r=6",
        "Cyl z: r=7",
        "GQuadric: {0,1,2} {3,4,5} {6,7,8} 9"};
    // clang-format on
    EXPECT_VEC_EQ(expected_strings, strings);
}
