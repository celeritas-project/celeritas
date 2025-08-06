//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/AllGeoTypedTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "orange/OrangeTestBase.hh"
#if CELERITAS_USE_VECGEOM
#    include "geocel/vg/VecgeomTestBase.hh"
#endif
#if CELERITAS_USE_GEANT4
#    include "geocel/g4/GeantGeoTestBase.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Type-parameterized geometry test harness.
 *
 * \tparam HP Geometry host Params class
 *
 * To use this class to test all available geometry types, add
 * \code
 *   ${_needs_geo} LINK_LIBRARIES ${_all_geo_libs}
 * \endcode
 *
 * to the \c celeritas_add_test argument in CMakeLists.txt, and instantiate all
 * the types with:
 *
 \code
  template<class HP>
  class MyFooTest : public AllGeoTypedTestBase<HP>
  {
    std::string_view gdml_basename() const final { return "simple-cms"; }
  };

  TYPED_TEST_SUITE(MyFooTest, AllGeoTestingTypes, AllGeoTestingTypeNames);

  TYPED_TEST(MyFooTest, bar)
  {
      using GeoTrackView = typename TestFixture::GeoTrackView;
      auto geo = this->geometry();
      auto track = this->make_track_view({1, 2, 3}, {0, 0, 1});
  }
 \endcode
 */
template<class HP>
class AllGeoTypedTestBase : public GenericGeoTestBase<HP>
{
  public:
    static void SetUpTestCase() { LazyGeantGeoManager::clear_lazy_geo(); }
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using AllGeoTestingTypes = ::testing::Types<
#if CELERITAS_USE_VECGEOM
    VecgeomParams,
#endif
#if CELERITAS_USE_GEANT4 && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
    GeantGeoParams,
#endif
    OrangeParams>;

//! Helper class for returning type names
struct AllGeoTestingTypeNames
{
    template<class U>
    static std::string GetName(int)
    {
        return GeoTraits<U>::name;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
