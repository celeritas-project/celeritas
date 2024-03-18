//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/grid/XsCalculator.hh"
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
    return std::make_shared<WentzelVIMscParams>(
        particles, materials, data.msc_models);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and material properties.
 */
WentzelVIMscParams::WentzelVIMscParams(ParticleParams const& particles,
                                       MaterialParams const& materials,
                                       VecImportMscModel const& mdata_vec)
{
    using units::MevEnergy;

    ScopedMem record_mem("WentzelVIMscParams.construct");

    HostVal<WentzelVIMscData> host_data;

    detail::MscParamsHelper helper(
        particles, materials, mdata_vec, ImportModelClass::wentzel_vi_uni);
    helper.build_ids(&host_data.ids);
    helper.build_xs(&host_data.xs, &host_data.reals);

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Save high/low energy limits
    XsCalculator calc_xs(host_data.xs[ItemId<XsGridData>(0)],
                         make_const_ref(host_data).reals);
    host_data.params.low_energy_limit = calc_xs.energy_min();
    host_data.params.high_energy_limit = calc_xs.energy_max();

    CELER_ASSERT(host_data);

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<WentzelVIMscData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
