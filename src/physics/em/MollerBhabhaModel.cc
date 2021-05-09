//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
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
                   << "missing electron and/or positron particles "
                      "(required for "
                   << this->label() << ")");

    interface_.electron_mass_c_sq
        = particles.get(interface_.electron_id).mass().value(); // [MeV]

    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MollerBhabhaModel::applicability() const -> SetApplicability
{
    Applicability electron_applic, positron_applic;

    // The electron applicability.lower is twice the one for positrons due to
    // its maximum transferable energy fraction being 0.5 (which is 1/2 the
    // positron's). This prevents it to run an infinite number of Moller
    // sampling loops.
    electron_applic.particle = interface_.electron_id;
    electron_applic.lower = units::MevEnergy{2 * interface_.min_valid_energy()};
    electron_applic.upper = units::MevEnergy{interface_.max_valid_energy()};

    positron_applic.particle = interface_.positron_id;
    positron_applic.lower    = units::MevEnergy{interface_.min_valid_energy()};
    positron_applic.upper    = electron_applic.upper;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void MollerBhabhaModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractRefs<MemSpace::device>& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::moller_bhabha_interact(interface_, pointers);
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
