//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Model and particles IDs.
 */
struct MollerBhabhaIds
{
    ModelId    model;
    ParticleId electron;
    ParticleId positron;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids.model && ids.electron && ids.gamma && inv_electron_mass > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MollerBhabhaData
{
    //! Electron mass * c^2 [MeV]
    real_type electron_mass_c_sq;

    //! Model's mininum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type min_valid_energy()
    {
        return 1e-3;
    }
    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type max_valid_energy()
    {
        return 100e6;
    }

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass_c_sq > 0;
    }
};

using MollerBhabhaHostRef   = MollerBhabhaData;
using MollerBhabhaDeviceRef = MollerBhabhaData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
