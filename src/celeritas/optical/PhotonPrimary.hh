//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhotonPrimary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Optical photon data used to initialize a photon track state.
 */
struct PhotonPrimary
{
    units::MevEnergy energy;
    Real3 position{0, 0, 0};
    Real3 direction{0, 0, 0};
    real_type time{};
    VolumeId volume{};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
