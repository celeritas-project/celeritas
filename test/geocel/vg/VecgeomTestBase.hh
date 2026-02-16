//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/cont/Span.hh"
#include "geocel/GenericGeoTestBase.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomGeoTraits.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class VecgeomTestBase : public GenericGeoTestBase<VecgeomParams>
{
    using Base = GenericGeoTestBase<VecgeomParams>;

  public:
    using SpanStringView = Span<std::string_view const>;

    // Keep track of log messages during load
    virtual SpanStringView expected_log_levels() const;

    // Construct via persistent geant_geo; see LazyGeantGeoManager
    SPConstGeo build_geometry() const override;

    // Update a checked track view to modify normal checking
    CheckedGeoTrackView make_checked_track_view() override;

    // Get the safety tolerance: lower for surface geo
    GenericGeoTrackingTolerance tracking_tol() const override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
