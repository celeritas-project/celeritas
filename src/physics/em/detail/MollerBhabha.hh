//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
template<MemSpace M>
struct ModelInteractRefs;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MollerBhabhaPointers
{
    //! Model ID
    ModelId model_id;

    //! ID of an electron
    ParticleId electron_id;
    //! ID of a positron
    ParticleId positron_id;
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
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && positron_id && electron_mass_c_sq > 0;
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch Moller-Bhabha interaction
void moller_bhabha_interact(
    const MollerBhabhaPointers&                device_pointers,
    const ModelInteractRefs<MemSpace::device>& interaction);

void moller_bhabha_interact(const MollerBhabhaPointers& device_pointers,
                            const ModelInteractRefs<MemSpace::host>& interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
