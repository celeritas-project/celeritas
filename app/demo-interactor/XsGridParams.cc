//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsGridParams.cc
//---------------------------------------------------------------------------//
#include "XsGridParams.hh"

#include <algorithm>
#include <cmath>
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "comm/Device.hh"
#include "physics/grid/UniformGrid.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input data.
 */
XsGridParams::XsGridParams(const Input& input)
    : prime_index_(
        std::find(input.energy.begin(), input.energy.end(), input.prime_energy)
        - input.energy.begin())
    , host_xs_(input.xs)
{
    CELER_EXPECT(input.energy.size() >= 2);
    CELER_EXPECT(input.energy.front() > 0);
    CELER_EXPECT(std::is_sorted(input.energy.begin(), input.energy.end()));
    CELER_EXPECT(input.xs.size() == input.energy.size());
    CELER_EXPECT(std::all_of(
        input.xs.begin(), input.xs.end(), [](real_type v) { return v >= 0; }));
    CELER_EXPECT(prime_index_ != input.energy.size());

    // Calculate uniform-in-logspace energy grid
    log_energy_ = celeritas::UniformGridData::from_bounds(
        std::log(input.energy.front()),
        std::log(input.energy.back()),
        input.energy.size());

#if CELERITAS_DEBUG
    {
        // Test soft equivalence between log energy grid and input energy to
        // make sure all the points are uniformly spaced
        celeritas::SoftEqual<> soft_eq(1e-8);
        celeritas::UniformGrid log_energy(log_energy_);
        CELER_ASSERT(log_energy.size() == input.energy.size());
        for (auto i : celeritas::range(input.energy.size()))
        {
            CELER_EXPECT(soft_eq(std::log(input.energy[i]), log_energy[i]));
        }
    }
#endif

    if (celeritas::is_device_enabled())
    {
        // Copy xs values to device
        xs_ = celeritas::DeviceVector<real_type>(input.xs.size());
        xs_.copy_to_device(celeritas::make_span(input.xs));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Access on-device data.
 */
auto XsGridParams::device_pointers() const -> XsGridPointers
{
    XsGridPointers result;
    result.log_energy  = log_energy_;
    result.prime_index = prime_index_;
    result.value       = xs_.device_pointers();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access on-host data.
 */
auto XsGridParams::host_pointers() const -> XsGridPointers
{
    XsGridPointers result;
    result.log_energy  = log_energy_;
    result.prime_index = prime_index_;
    result.value       = celeritas::make_span(host_xs_);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
