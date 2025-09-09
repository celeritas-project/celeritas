//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Applicability.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <tuple>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Range where a model and/or process is valid.
 *
 * This class is used during setup for specifying the ranges of applicability
 * for a physics model or process. The interval is *closed* on the lower energy
 * range and *open* on the upper energy. So a threshold reaction should have
 * the lower energy set to the threshold.
 *
 * An unset value for "material" means it applies to all materials; however,
 * the particle ID should always be set.
 */
struct Applicability
{
    using EnergyUnits = units::Mev;
    using Energy = RealQuantity<EnergyUnits>;

    PhysMatId material{};
    ParticleId particle{};
    Energy lower = zero_quantity();
    Energy upper = max_quantity();

    //! Whether applicability is in a valid state
    inline explicit operator bool() const
    {
        return static_cast<bool>(particle) && lower < upper;
    }
};

//!@{
//! Comparators
inline bool operator==(Applicability const& lhs, Applicability const& rhs)
{
    return std::make_tuple(lhs.material, lhs.particle, lhs.lower, lhs.upper)
           == std::make_tuple(rhs.material, rhs.particle, rhs.lower, rhs.upper);
}

inline bool operator<(Applicability const& lhs, Applicability const& rhs)
{
    return std::make_tuple(lhs.material, lhs.particle, lhs.lower, lhs.upper)
           < std::make_tuple(rhs.material, rhs.particle, rhs.lower, rhs.upper);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
