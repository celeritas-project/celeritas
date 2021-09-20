//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/material/Types.hh"

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
    ItemRange<AtomicRelaxTransition> transitions;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental atomic relaxation data.
 */
struct AtomicRelaxElement
{
    ItemRange<AtomicRelaxSubshell> shells;

    size_type max_secondary;  //!< Maximum number of secondaries possible
    size_type max_stack_size; //!< Maximum size of the subshell vacancy stack

    //! Check whether the element is assigned (false for Z < 6).
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !shells.empty() && max_secondary > 0 && max_stack_size > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Electron subshell transition data for atomic relaxation.
 */
template<Ownership W, MemSpace M>
struct AtomicRelaxData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Items<AtomicRelaxTransition>     transitions;
    Items<AtomicRelaxSubshell>       shells;
    ElementItems<AtomicRelaxElement> elements;
    ParticleId                       electron_id;
    ParticleId                       gamma_id;

    //// MEMBER FUNCTIONS ////

    //! Check whether the interface is assigned.
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !transitions.empty() && !shells.empty() && !elements.empty()
               && electron_id && gamma_id;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AtomicRelaxData& operator=(const AtomicRelaxData<W2, M2>& other)
    {
        transitions = other.transitions;
        shells      = other.shells;
        elements    = other.elements;
        electron_id = other.electron_id;
        gamma_id    = other.gamma_id;
        return *this;
    }
};

using AtomicRelaxPointers
    = AtomicRelaxData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
} // namespace celeritas
