//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTestBase.cc
//---------------------------------------------------------------------------//
#include "VecgeomTestBase.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GenericGeoTestBase.t.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct via persistent geant_geo; see LazyGeantGeoManager.
 */
auto VecgeomTestBase::build_geometry() const -> SPConstGeo
{
    using std::cout;
    using std::endl;

    cout << color_code('x') << "VecGeom v" << Version::from_package("VecGeom")
         << " (" << ::celeritas::cmake::vecgeom_options << ") using G4VG v"
         << Version::from_package("G4VG") << " and Geant4 v"
         << Version::from_package("Geant4") << color_code(' ') << endl;

    ScopedLogStorer scoped_log_{&celeritas::world_logger(), LogLevel::warning};
    auto result = Base::build_geometry();
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Update a checked track view to modify normal checking.
 */
CheckedGeoTrackView VecgeomTestBase::make_checked_track_view()
{
    auto result = GenericGeoTestBase<VecgeomParams>::make_checked_track_view();
    result.check_normal(false);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the safety tolerance: lower for surface geo.
 */
GenericGeoTrackingTolerance VecgeomTestBase::tracking_tol() const
{
    GenericGeoTrackingTolerance result = Base::tracking_tol();
    return result;
}

//---------------------------------------------------------------------------//
template class GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
