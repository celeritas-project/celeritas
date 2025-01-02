//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
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

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find the index of the distribution from which to generate the primary.
 *
 * This finds the index in offsets for which offsets[result - 1] <= value <
 * offsets[result].
 */
inline CELER_FUNCTION size_type find_distribution_index(Span<size_type> offsets,
                                                        size_type value)
{
    CELER_EXPECT(!offsets.empty());

    auto iter = celeritas::lower_bound(offsets.begin(), offsets.end(), value);
    CELER_ASSERT(iter != offsets.end());

    if (value == *iter)
    {
        ++iter;
    }
    return iter - offsets.begin();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
