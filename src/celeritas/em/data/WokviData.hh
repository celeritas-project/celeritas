//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WokviData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

struct WokviIds
{
    ActionId action;
    ParticleId electron;
    ParticleId positron;
    ParticleId proton;

    explicit CELER_FUNCTION operator bool() const
    {
        return action && electron && positron && proton;
    }
};

struct WokviElementData
{
    real_type screen_r_sq;
    real_type screen_r_sq_elec;
    real_type form_factor;
    real_type inv_a23;
};

template<Ownership W, MemSpace M>
struct WokviData
{
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;

    // Ids
    WokviIds ids;

    // Per element form factors
    ElementItems<WokviElementData> elem_data;

    // Other parameters
    real_type factor_A2;  // 0.5 * (angleLimitFactor * hbarc / fermi)^2
    real_type factor_B1;  // 0.5 * pi * alpha^2
    real_type coeff;  // 2 pi (e_mass * e_radius)^2
    real_type electron_mass;

    explicit CELER_FUNCTION operator bool() const
    {
        return ids && !elem_data.empty();
    }

    template<Ownership W2, MemSpace M2>
    WokviData& operator=(WokviData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        elem_data = other.elem_data;
        factor_A2 = other.factor_A2;
        factor_B1 = other.factor_B1;
        coeff = other.coeff;
        electron_mass = other.electron_mass;
        return *this;
    }
};

using WokviDeviceRef = DeviceCRef<WokviData>;
using WokviHostRef = HostCRef<WokviData>;
using WokviRef = NativeCRef<WokviData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
