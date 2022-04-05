//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElectronBremsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "physics/base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! IDs used by brems
struct ElectronBremIds
{
    //! Model ID
    ActionId action;
    //! ID of a gamma
    ParticleId gamma;
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action && gamma && electron && positron;
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
