//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/WentzelVIMscParams.cc
//---------------------------------------------------------------------------//
#include "WentzelVIMscParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/data/WentzelVIMscData.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct if Wentzel VI model is present, or else return nullptr.
 */
std::shared_ptr<WentzelVIMscParams>
WentzelVIMscParams::from_import(ParticleParams const& particles,
                                MaterialParams const& materials,
                                ImportData const& data)
{
    auto is_wentzel = [](ImportMscModel const& imm) {
        return imm.model_class == ImportModelClass::wentzel_vi_uni;
    };
    if (!std::any_of(data.msc_models.begin(), data.msc_models.end(), is_wentzel))
    {
        // No WentzelVI MSC present
        return nullptr;
    }

    Options opts;
    opts.lambda_limit = data.em_params.msc_lambda_limit;
    opts.safety_fact = data.em_params.msc_safety_factor;
    opts.range_fact = data.em_params.msc_range_factor;
    opts.geom_fact = data.em_params.msc_geom_factor;

    return std::make_shared<WentzelVIMscParams>(
        particles, materials, data.msc_models, opts);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and material properties.
 */
WentzelVIMscParams::WentzelVIMscParams(ParticleParams const& particles,
                                       MaterialParams const& materials,
                                       VecImportMscModel const& mdata_vec,
                                       Options options)
{
    using units::MevEnergy;

    ScopedMem record_mem("WentzelVIMscParams.construct");

    HostVal<WentzelVIMscData> host_data;

    this->build_ids(&host_data.ids, particles);
    this->build_parameters(&host_data.msc_params, options);
    this->build_xs(&host_data.xs,
                   &host_data.reals,
                   particles,
                   materials,
                   mdata_vec,
                   ImportModelClass::wentzel_vi_uni);

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Save high/low energy limits
    auto const& grid = host_data.xs[ItemId<XsGridData>(0)].log_energy;
    host_data.params.low_energy_limit = MevEnergy(std::exp(grid.front));
    host_data.params.high_energy_limit = MevEnergy(std::exp(grid.back));

    CELER_ASSERT(host_data);

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<WentzelVIMscData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
