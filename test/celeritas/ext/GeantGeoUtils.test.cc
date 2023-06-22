//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantGeoUtils.hh"

#include <G4LogicalVolume.hh>

#include "celeritas/ext/GeantGeoParams.hh"

#include "../GenericGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
// Get volume names for a bunch of G4LV*
template<class InputIterator>
decltype(auto) get_vol_names(InputIterator iter, InputIterator stop)
{
    std::vector<std::string> result;
    for (; iter != stop; ++iter)
    {
        CELER_ASSERT(*iter);
        result.push_back((*iter)->GetName());
    }
    return result;
}
}  // namespace
//---------------------------------------------------------------------------//

class GeantGeoUtilsTest : public GenericGeantGeoTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    void SetUp() override
    {
        // Build geometry during setup
        ASSERT_TRUE(this->geometry());
    }
};

class SolidsTest : public GeantGeoUtilsTest
{
    std::string geometry_basename() const override { return "solids"; }
};

using FindGeantVolumesTest = SolidsTest;

TEST_F(FindGeantVolumesTest, standard)
{
    static std::string_view const labels[] = {"box500", "trd3", "trd1"};
    auto vols = find_geant_volumes(make_span(labels));
    auto vol_names = get_vol_names(vols.begin(), vols.end());
    static char const* const expected_vol_names[] = {"box500", "trd1", "trd3"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);
}

TEST_F(FindGeantVolumesTest, missing)
{
    static std::string_view const labels[] = {"box500", "trd3", "turd3"};
    EXPECT_THROW(find_geant_volumes(make_span(labels)), RuntimeError);
}

TEST_F(FindGeantVolumesTest, duplicate)
{
    static std::string_view const labels[] = {"trd3_refl"};
    auto vols = find_geant_volumes(make_span(labels));
    auto vol_names = get_vol_names(vols.begin(), vols.end());
    static char const* const expected_vol_names[] = {"trd3_refl", "trd3_refl"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
