//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaModel.cc
//---------------------------------------------------------------------------//
#include "MollerBhabhaModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MollerBhabhaModel::MollerBhabhaModel(ModelId               id,
                                     const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.electron_id = particles.find(pdg::electron());
    interface_.positron_id = particles.find(pdg::positron());

    CELER_VALIDATE(interface_.electron_id && interface_.positron_id,
                   "Electron and positron particles must be enabled to use "
                   "the Moller-Bhabha Model.");
    interface_.electron_mass_c_sq
        = particles.get(interface_.electron_id).mass.value()
          * units::CLightSq().value();

    interface_.min_valid_energy_ = units::MevEnergy{1e-3};

    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MollerBhabhaModel::applicability() const -> SetApplicability
{
    Applicability electron_applic;
    electron_applic.particle = interface_.electron_id;
    electron_applic.lower    = units::MevEnergy{1e-3};  // TODO: double-check
    electron_applic.upper    = units::MevEnergy{100e6}; // TODO: double-check

    return {electron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void MollerBhabhaModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractPointers& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::klein_nishina_interact(interface_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId MollerBhabhaModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
