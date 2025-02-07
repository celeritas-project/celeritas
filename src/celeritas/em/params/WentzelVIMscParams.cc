//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/WentzelVIMscParams.cc
//---------------------------------------------------------------------------//
#include "WentzelVIMscParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/params/detail/MscParamsHelper.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct if Wentzel VI model is present, or else return nullptr.
 */
std::shared_ptr<WentzelVIMscParams>
WentzelVIMscParams::from_import(ParticleParams const& particles,
                                ImportData const& data)
{
    if (!has_msc_model(data, ImportModelClass::wentzel_vi_uni))
    {
        // No WentzelVI MSC present
        return nullptr;
    }
    return std::make_shared<WentzelVIMscParams>(particles, data.msc_models);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and material properties.
 */
WentzelVIMscParams::WentzelVIMscParams(ParticleParams const& particles,
                                       VecImportMscModel const& mdata_vec)
{
    using units::MevEnergy;

    ScopedMem record_mem("WentzelVIMscParams.construct");

    HostVal<WentzelVIMscData> host_data;

    detail::MscParamsHelper helper(
        particles, mdata_vec, ImportModelClass::wentzel_vi_uni);
    helper.build_ids(&host_data.ids, &host_data.pid_to_xs);
    helper.build_xs(&host_data.xs, &host_data.reals);
    host_data.num_particles = helper.particle_ids().size();

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Number of applicable particles
    host_data.num_particles = helper.particle_ids().size();

    // Get the cross section energy grid limits (this checks that the limits
    // are the same for all particles/materials)
    auto energy_limit = helper.energy_grid_bounds();
    host_data.params.low_energy_limit = energy_limit[0];
    host_data.params.high_energy_limit = energy_limit[1];

    CELER_ASSERT(host_data);

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<WentzelVIMscData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
