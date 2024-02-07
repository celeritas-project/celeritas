//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPrimary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Optical photon data used to initialize a photon track state.
 */
struct OpticalPrimary
{
    units::MevEnergy energy;
    Real3 position{0, 0, 0};
    Real3 direction{0, 0, 0};
    Real3 polarization{0, 0, 0};
    real_type time{};
    VolumeId volume{};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
