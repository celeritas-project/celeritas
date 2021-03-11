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
 *
 * \todo When multiple methods that use atomic relaxation are in place, we
 * should share AtomicRelaxationParams among them, and move
 * `RelaxationScratchData` into that class, to reduce fixed-size memory
 * allocations.
 */
class LivermorePEModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    LivermorePEModel(ModelId                  id,
                     const ParticleParams&    particles,
                     const LivermorePEParams& data);

    // Construct with transition data for atomic relaxation
    LivermorePEModel(ModelId                       id,
                     const ParticleParams&         particles,
                     const LivermorePEParams&      data,
                     const AtomicRelaxationParams& atomic_relaxation,
                     size_type                     num_vacancies);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const ModelInteractPointers&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Livermore photoelectric"; }

    // Access data on device
    inline const detail::LivermorePEPointers& device_pointers() const;

  private:
    detail::LivermorePEPointers     interface_;
    detail::RelaxationScratchData<Ownership::value, MemSpace::device>
        relax_scratch_;
    detail::RelaxationScratchData<Ownership::reference, MemSpace::device>
        relax_scratch_ref_;
};

//---------------------------------------------------------------------------//
/*!
 * Access data on device.
 */
const detail::LivermorePEPointers& LivermorePEModel::device_pointers() const
{
    return interface_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
