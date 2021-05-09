//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerModel.cc
//---------------------------------------------------------------------------//
#include "BetheHeitlerModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
BetheHeitlerModel::BetheHeitlerModel(ModelId               id,
                                     const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.electron_id = particles.find(pdg::electron());
    interface_.positron_id = particles.find(pdg::positron());
    interface_.gamma_id    = particles.find(pdg::gamma());

    CELER_VALIDATE(interface_.electron_id && interface_.positron_id
                       && interface_.gamma_id,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->label() << ")");
    interface_.inv_electron_mass
        = 1 / particles.get(interface_.electron_id).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BetheHeitlerModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = interface_.gamma_id;
    photon_applic.lower    = units::MevEnergy{1.5};
    photon_applic.upper    = units::MevEnergy{1e5};

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void BetheHeitlerModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractRefs<MemSpace::device>& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::bethe_heitler_interact(interface_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId BetheHeitlerModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
