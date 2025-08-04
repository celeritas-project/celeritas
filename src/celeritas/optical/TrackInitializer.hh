//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackInitializer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical photon data used to initialize a photon track state.
 */
struct TrackInitializer
{
    units::MevEnergy energy;
    Real3 position{0, 0, 0};
    Real3 direction{0, 0, 0};
    Real3 polarization{0, 0, 0};
    real_type time{};
    ImplVolumeId volume{};
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
