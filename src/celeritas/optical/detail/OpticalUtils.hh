//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a native-system wavelength into a photon energy.
 */
inline CELER_FUNCTION units::MevEnergy
wavelength_to_energy(real_type wavelength)
{
    CELER_EXPECT(wavelength > 0);
    return native_value_to<units::MevEnergy>(
        (constants::h_planck * constants::c_light) / wavelength);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a photon energy to native-system wavelength.
 */
inline CELER_FUNCTION real_type energy_to_wavelength(units::MevEnergy energy)
{
    CELER_EXPECT(energy > zero_quantity());
    return (constants::h_planck * constants::c_light)
           / native_value_from(energy);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
