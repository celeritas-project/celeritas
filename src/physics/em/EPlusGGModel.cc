//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGModel.cc
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/em/generated/EPlusGGInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
EPlusGGModel::EPlusGGModel(ModelId id, const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.positron_id = particles.find(pdg::positron());
    interface_.gamma_id    = particles.find(pdg::gamma());

    CELER_VALIDATE(interface_.positron_id && interface_.gamma_id,
                   << "missing positron and/or gamma particles (required for "
                   << this->label() << ")");
    interface_.electron_mass
        = particles.get(interface_.positron_id).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 *
 * \todo The Interactor allows non-rest energies as does the
 * G4eeToTwoGammaModel, which also defines ComputeCrossSectionPerAtom, but
 * does not seem to export cross sections. Is it because they're generated on
 * the fly?
 */
auto EPlusGGModel::applicability() const -> SetApplicability
{
    Applicability in_flight;
    in_flight.particle = interface_.positron_id;
    in_flight.lower    = zero_quantity();
    in_flight.upper    = units::MevEnergy{1e8}; // 100 TeV

    return {Applicability::at_rest(interface_.positron_id), in_flight};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void EPlusGGModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractRefs<MemSpace::device>& pointers) const
{
#if CELERITAS_USE_CUDA
    generated::eplusgg_interact(interface_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

void EPlusGGModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractRefs<MemSpace::host>& pointers) const
{
    generated::eplusgg_interact(interface_, pointers);
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId EPlusGGModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
