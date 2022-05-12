//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ApplyCutoffModel.cc
//---------------------------------------------------------------------------//
#include "ApplyCutoffModel.hh"

#include <algorithm>
#include <limits>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/CutoffParams.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and list of particles to kill.
 */
ApplyCutoffModel::ApplyCutoffModel(ActionId id, SPConstCutoff cutoffs)
    : cutoffs_(std::move(cutoffs))
{
    CELER_EXPECT(id);
    CELER_EXPECT(cutoffs_);

    data_.ids.action = id;
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 *
 * \todo Currently the physics doesn't support per-material applicability; so
 * we just take the minimum energy over all materials as the cutoff.
 */
auto ApplyCutoffModel::applicability() const -> SetApplicability
{
    using MevEnergy             = celeritas::units::MevEnergy;
    const CutoffParams&    cuts = *cutoffs_;
    std::vector<real_type> particle_energies(
        cuts.num_particles(), std::numeric_limits<real_type>::max());

    SetApplicability result;
    for (auto mid : range(MaterialId{cuts.num_materials()}))
    {
        auto cut_view = cuts.get(mid);
        for (auto pidx : range(cuts.num_particles()))
        {
            particle_energies[pidx]
                = std::min(particle_energies[pidx],
                           cut_view.energy(ParticleId{pidx}).value());
        }
    }
    for (auto pidx : range(cuts.num_particles()))
    {
        result.insert(
            celeritas::Applicability{MaterialId{},
                                     ParticleId{pidx},
                                     MevEnergy{1e-9},
                                     MevEnergy{particle_energies[pidx]}});
    }
    return result;
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void ApplyCutoffModel::execute(CoreDeviceRef const& core) const
{
    generated::apply_cutoff_interact(data_, core);
}

void ApplyCutoffModel::execute(CoreHostRef const& core) const
{
    generated::apply_cutoff_interact(data_, core);
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
