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
    auto wentzel = find_msc_models(data, ImportModelClass::wentzel_vi_uni);
    if (wentzel.empty())
    {
        // No WentzelVI MSC present
        return nullptr;
    }

    Options opts;
    // Use combined single and multiple Coulomb scattering if both the single
    // scattering and the Wentzel VI models are present
    opts.is_combined
        = !find_models(data, ImportModelClass::e_coulomb_scattering).empty();
    opts.polar_angle_limit = wentzel.front()->params.polar_angle_limit;
    CELER_ASSERT(std::all_of(
        wentzel.begin(), wentzel.end(), [&opts](ImportMscModel const* m) {
            return m->params.polar_angle_limit == opts.polar_angle_limit;
        }));
    opts.screening_factor = data.em_params.screening_factor;
    opts.angle_limit_factor = data.em_params.angle_limit_factor;

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

    detail::MscParamsHelper helper(
        particles, materials, mdata_vec, ImportModelClass::wentzel_vi_uni);
    helper.build_ids(&host_data.ids);
    helper.build_xs(&host_data.xs, &host_data.reals);

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Whether to use combined single and multiple scattering
    host_data.coulomb_params.is_combined = options.is_combined;

    // Maximum scattering polar angle
    host_data.coulomb_params.costheta_limit
        = options.is_combined ? std::cos(options.polar_angle_limit) : -1;

    host_data.coulomb_params.a_sq_factor
        = real_type(0.5)
          * ipow<2>(options.angle_limit_factor * constants::hbar_planck
                    * constants::c_light * 1e-15 * units::meter);
    host_data.coulomb_params.screening_factor = options.screening_factor;

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
