//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEModel.cc
//---------------------------------------------------------------------------//
#include "LivermorePEModel.hh"

#include "base/Assert.hh"
#include "comm/Device.hh"
#include "physics/base/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
LivermorePEModel::LivermorePEModel(ModelId                  id,
                                   const ParticleParams&    particles,
                                   const LivermorePEParams& data)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.electron_id = particles.find(pdg::electron());
    interface_.gamma_id    = particles.find(pdg::gamma());
    interface_.data        = data.device_pointers();

    CELER_VALIDATE(interface_.electron_id && interface_.gamma_id,
                   "Electron and gamma particles must be enabled to use the "
                   "Livermore Photoelectric Model.");
    interface_.inv_electron_mass
        = 1 / particles.get(interface_.electron_id).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with transition data for atomic relaxation.
 */
LivermorePEModel::LivermorePEModel(
    ModelId                       id,
    const ParticleParams&         particles,
    const LivermorePEParams&      data,
    const AtomicRelaxationParams& atomic_relaxation,
    size_type                     num_vacancies)
    : LivermorePEModel(id, particles, data)
{
    CELER_EXPECT(num_vacancies > 0);
    resize(&relax_scratch_.vacancies, num_vacancies);
    relax_scratch_ref_ = relax_scratch_;

    interface_.atomic_relaxation = atomic_relaxation.device_pointers();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto LivermorePEModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = interface_.gamma_id;
    photon_applic.lower    = zero_quantity();
    photon_applic.upper    = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void LivermorePEModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractPointers& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::livermore_pe_interact(
        this->device_pointers(), relax_scratch_ref_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId LivermorePEModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
