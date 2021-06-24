//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlungModel.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuBremsstrahlungModel::MuBremsstrahlungModel(ModelId               id,
                                             const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.gamma_id    = particles.find(pdg::gamma());
    interface_.mu_minus_id = particles.find(pdg::mu_minus());
    interface_.mu_plus_id  = particles.find(pdg::mu_plus());

    CELER_VALIDATE(interface_.gamma_id && interface_.mu_minus_id
                   && interface_.mu_plus_id,
                   << "missing muon and/or gamma particles "
                      "(required for "
                   << this->label() << ")");

    interface_.electron_mass = 
                        particles.get(particles.find(pdg::electron())).mass();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MuBremsstrahlungModel::applicability() const -> SetApplicability
{
    Applicability mu_minus_applic, mu_plus_applic;

    mu_minus_applic.particle = interface_.mu_minus_id;
    mu_minus_applic.lower    = units::MevEnergy{0.2};
    mu_minus_applic.upper    = units::MevEnergy{1e3};

    mu_plus_applic.particle = interface_.mu_plus_id;
    mu_plus_applic.lower    = units::MevEnergy{0.2};
    mu_plus_applic.upper    = units::MevEnergy{1e3};

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void MuBremsstrahlungModel::interact(
    CELER_MAYBE_UNUSED const  ModelInteractRefs<MemSpace::device>& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::mu_bremsstrahlung_interact(interface_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId MuBremsstrahlungModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

