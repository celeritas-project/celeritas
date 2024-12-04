//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/AtomicRelaxationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Atomic relaxation transition data. The transition probabilities describe
 * both radiative and non-radiative transitions.
 */
struct AtomicRelaxTransition
{
    SubshellId initial_shell;  //!< Index of the originating shell
    SubshellId auger_shell;  //!< Index of the Auger electron shell
    real_type probability;
    units::MevEnergy energy;
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

    //! Check whether the element is assigned (false for Z < 6).
    explicit CELER_FUNCTION operator bool() const
    {
        return !shells.empty() && max_secondary > 0;
    }
};

struct AtomicRelaxIds
{
    ParticleId electron;
    ParticleId gamma;

    //! Check whether IDs are assigned
    explicit CELER_FUNCTION operator bool() const { return electron && gamma; }
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

    AtomicRelaxIds ids;
    Items<AtomicRelaxTransition> transitions;
    Items<AtomicRelaxSubshell> shells;
    ElementItems<AtomicRelaxElement> elements;
    size_type max_stack_size{};

    //// MEMBER FUNCTIONS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && !transitions.empty() && !shells.empty()
               && !elements.empty() && max_stack_size > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AtomicRelaxParamsData& operator=(AtomicRelaxParamsData<W2, M2> const& other)
    {
        ids = other.ids;
        transitions = other.transitions;
        shells = other.shells;
        elements = other.elements;
        max_stack_size = other.max_stack_size;
        return *this;
    }
};

using AtomicRelaxParamsRef = NativeCRef<AtomicRelaxParamsData>;

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
    Items<SubshellId> scratch;  // 2D array: [num states][max stack size]
    size_type num_states;

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
        scratch = other.scratch;
        num_states = other.num_states;
        return *this;
    }
};

using AtomicRelaxStateRef = NativeRef<AtomicRelaxStateData>;

//---------------------------------------------------------------------------//
/*!
 * Resize state data in host code.
 */
template<MemSpace M>
inline void resize(AtomicRelaxStateData<Ownership::value, M>* state,
                   HostCRef<AtomicRelaxParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&state->scratch, size * params.max_stack_size);
    state->num_states = size;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
