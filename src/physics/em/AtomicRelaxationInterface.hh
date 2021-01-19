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
 * Atomic relaxation transition data.
 *
 * The size of the arrays is the number of possible transitions that can occur
 * in which an electron from an upper shell drops down into this shell. The
 * transition probabilities describe both radiative and non-radiative
 * transitions. If no Auger electron shell is provided for a particular
 * transition, it is a radiative transition; otherwise it is a non-radiative
 * transition.
 */
struct AtomicRelaxSubshell
{
    Span<size_type> initial_shell; //!< Index of the originating shell
    Span<size_type> auger_shell;   //!< Index of the Auger electron shell
    Span<real_type> transition_energy;
    Span<real_type> transition_prob;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental atomic relaxation data.
 */
struct AtomicRelaxElement
{
    Span<const AtomicRelaxSubshell> shells;

    size_type max_secondaries; //!< Maximum number of secondaries possible

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
    size_type                      unassigned; //!< Flag for nassigned shell id

    //! Check whether the interface is assigned.
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !elements.empty() && electron_id && gamma_id;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
