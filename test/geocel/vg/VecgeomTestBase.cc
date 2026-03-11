//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTestBase.cc
//---------------------------------------------------------------------------//
#include "VecgeomTestBase.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/ColorUtils.hh"
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
    using namespace celeritas::cmake;
    using std::cout;
    using std::endl;

    cout << color_code('x') << "VecGeom v" << vecgeom_version << " ("
         << vecgeom_options << ") using G4VG v" << g4vg_version
         << " and Geant4 v" << geant4_version << color_code(' ') << endl;

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

    if (CELERITAS_VECGEOM_SURFACE)
    {
        result.safety = 6e-5;
    }
    return result;
}

//---------------------------------------------------------------------------//
template class GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
