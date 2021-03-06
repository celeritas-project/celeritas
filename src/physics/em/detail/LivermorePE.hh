//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/StackAllocatorInterface.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/em/AtomicRelaxationInterface.hh"
#include "physics/em/LivermorePEInterface.hh"

namespace celeritas
{
struct ModelInteractPointers;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a LivermorePEInteractor.
 *
 * \todo Template on MemSpace.
 */
struct LivermorePEPointers
{
    //! Model ID
    ModelId model_id;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleId electron_id;
    //! ID of a gamma
    ParticleId gamma_id;
    //! Livermore EPICS2014 photoelectric data
    LivermorePEParamsPointers data;
    //! EADL transition data used for atomic relaxation
    AtomicRelaxParamsPointers atomic_relaxation;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && gamma_id
               && data;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Temporary data needed during interaction.
 */
template<Ownership W, MemSpace M>
struct RelaxationScratchData
{
    //! Storage for the stack of vacancy subshell IDs
    StackAllocatorData<SubshellId, W, M> vacancies;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return bool(vacancies); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    RelaxationScratchData& operator=(RelaxationScratchData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        vacancies = other.vacancies;
        return *this;
    }
};

using RelaxationScratchPointers
    = RelaxationScratchData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Livermore photoelectric interaction
void livermore_pe_interact(const LivermorePEPointers&       device_pointers,
                           const RelaxationScratchPointers& device_scratch,
                           const ModelInteractPointers&     interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
