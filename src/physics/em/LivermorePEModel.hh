//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/em/AtomicRelaxationParams.hh"
#include "physics/em/LivermorePEParams.hh"
#include "detail/LivermorePE.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Livermore photoelectric model interaction.
 */
class LivermorePEModel final : public Model
{
  public:
    //!@{
    //! Type aliases
    using SPConstSubshellIdAllocatorStore
        = std::shared_ptr<SubshellIdAllocatorStore>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    LivermorePEModel(ModelId                  id,
                     const ParticleParams&    particles,
                     const LivermorePEParams& data);

    // Construct with transition data for atomic relaxation
    LivermorePEModel(ModelId                         id,
                     const ParticleParams&           particles,
                     const LivermorePEParams&        data,
                     const AtomicRelaxationParams&   atomic_relaxation,
                     SPConstSubshellIdAllocatorStore vacancies);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const ModelInteractPointers&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Livermore photoelectric"; }

    // Access data on device
    detail::LivermorePEPointers device_pointers() const;

  private:
    detail::LivermorePEPointers     interface_;
    SPConstSubshellIdAllocatorStore vacancies_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
