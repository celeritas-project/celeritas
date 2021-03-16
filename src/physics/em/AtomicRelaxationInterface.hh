//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Atomic relaxation transition data. The transition probabilities describe
 * both radiative and non-radiative transitions.
 */
struct AtomicRelaxTransition
{
    SubshellId initial_shell; //!< Index of the originating shell
    SubshellId auger_shell;   //!< Index of the Auger electron shell
    real_type  probability;
    real_type  energy;
};

//---------------------------------------------------------------------------//
/*!
 * Electron subshell data.
 */
struct AtomicRelaxSubshell
{
    Span<const AtomicRelaxTransition> transitions;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental atomic relaxation data.
 */
struct AtomicRelaxElement
{
    Span<const AtomicRelaxSubshell> shells;

    size_type max_secondary;  //!< Maximum number of secondaries possible
    size_type max_stack_size; //!< Maximum size of the subshell vacancy stack

    //! Check whether the element is assigned (false for Z < 6).
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !shells.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Access atomic relaxation data on device.
 */
struct AtomicRelaxParamsPointers
{
    Span<const AtomicRelaxElement> elements;
    ParticleId                     electron_id;
    ParticleId                     gamma_id;
    units::MevEnergy               electron_cut; //!< Production thresholds
    units::MevEnergy               gamma_cut;

    //! Check whether the interface is assigned.
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !elements.empty() && electron_id && gamma_id;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
