//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
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
    size_type max_secondary; //!< Maximum number of secondaries possible

    //! Check whether the element is assigned (false for Z < 6).
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !shells.empty() && max_secondary > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Electron subshell transition data for atomic relaxation.
 */
template<Ownership W, MemSpace M>
struct AtomicRelaxParamsData
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
    size_type                        max_stack_size{};

    //// MEMBER FUNCTIONS ////

    //! Check whether the interface is assigned.
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !transitions.empty() && !shells.empty() && !elements.empty()
               && electron_id && gamma_id && max_stack_size > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AtomicRelaxParamsData& operator=(const AtomicRelaxParamsData<W2, M2>& other)
    {
        transitions    = other.transitions;
        shells         = other.shells;
        elements       = other.elements;
        electron_id    = other.electron_id;
        gamma_id       = other.gamma_id;
        max_stack_size = other.max_stack_size;
        return *this;
    }
};

using AtomicRelaxParamsPointers
    = AtomicRelaxParamsData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Temporary data needed during interaction.
 */
template<Ownership W, MemSpace M>
struct AtomicRelaxStateData
{
    template<class T>
    using Items = StateCollection<T, W, M>;

    //! Storage for the stack of vacancy subshell IDs
    Items<SubshellId> scratch; // 2D array: [num states][max stack size]
    size_type         num_states;

    //! Whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !scratch.empty() && num_states > 0;
    }

    //! State size
    CELER_FUNCTION size_type size() const { return num_states; }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    AtomicRelaxStateData& operator=(AtomicRelaxStateData<W2, M2>& other)
    {
        scratch    = other.scratch;
        num_states = other.num_states;
        return *this;
    }
};

using AtomicRelaxStatePointers
    = AtomicRelaxStateData<Ownership::reference, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Resize state data in host code.
 */
template<MemSpace M>
inline void
resize(AtomicRelaxStateData<Ownership::value, M>* data,
       const AtomicRelaxParamsData<Ownership::const_reference, MemSpace::host>&
                 params,
       size_type size)
{
    CELER_EXPECT(size > 0);
    make_builder(&data->scratch).resize(size * params.max_stack_size);
    data->num_states = size;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
