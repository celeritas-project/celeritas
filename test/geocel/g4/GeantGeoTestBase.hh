//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GenericGeoTestBase.hh"
#include "geocel/g4/GeantGeoData.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "geocel/g4/GeantGeoTrackView.hh"
#include "geocel/g4/GeantGeoTraits.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class GeantGeoTestBase : public GenericGeoTestBase<GeantGeoParams>
{
  public:
    //! Ignore the first N VolumeId due to global int shenanigans
    VolumeId::size_type volume_offset() const final
    {
        return this->geometry()->lv_offset();
    }

    //! Ignore the first N VolumeInstanceId due to global int shenanigans
    VolumeInstanceId::size_type volume_instance_offset() const final
    {
        return this->geometry()->pv_offset();
    }

    //! Get the world volume
    G4VPhysicalVolume const* g4world() const final
    {
        return this->geometry()->world();
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
