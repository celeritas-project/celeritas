//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/UrbanMscParams.cc
//---------------------------------------------------------------------------//
#include "UrbanMscParams.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/params/detail/MscParamsHelper.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct if Urban model is present, or else return nullptr.
 */
std::shared_ptr<UrbanMscParams>
UrbanMscParams::from_import(ParticleParams const& particles,
                            MaterialParams const& materials,
                            ImportData const& data)
{
    if (!has_msc_model(data, ImportModelClass::urban_msc))
    {
        // No Urban MSC present
        return nullptr;
    }
    return std::make_shared<UrbanMscParams>(
        particles, materials, data.msc_models);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and material properties.
 */
UrbanMscParams::UrbanMscParams(ParticleParams const& particles,
                               MaterialParams const& materials,
                               VecImportMscModel const& mdata_vec)
{
    using units::MevEnergy;
    using UrbanParMatId = UrbanMscParMatData::UrbanParMatId;
    using UPMT = UrbanParMatType;

    ScopedMem record_mem("UrbanMscParams.construct");

    HostVal<UrbanMscData> host_data;

    detail::MscParamsHelper helper(
        particles, mdata_vec, ImportModelClass::urban_msc);
    helper.build_ids(&host_data.ids, &host_data.pid_to_xs);
    helper.build_xs(&host_data.xs, &host_data.reals);

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Number of applicable particles
    host_data.num_particles = helper.particle_ids().size();
    CELER_ASSERT(host_data.num_particles >= 2);

    // Number of different particle categories in the particle and
    // material-dependent parameter data: electrons and positrons always, and
    // muons/hadrons when present
    host_data.num_par_mat = min<size_type>(
        host_data.num_particles, static_cast<size_type>(UPMT::size_));

    // Map from particle ID to index in particle and material-dependent data
    std::vector<UrbanParMatId> pid_to_pmdata(particles.size());
    for (auto par_id : helper.particle_ids())
    {
        CELER_ASSERT(par_id < pid_to_pmdata.size());
        if (par_id == host_data.ids.electron)
        {
            pid_to_pmdata[par_id.unchecked_get()]
                = UrbanParMatId(static_cast<size_type>(UPMT::electron));
        }
        else if (par_id == host_data.ids.positron)
        {
            pid_to_pmdata[par_id.unchecked_get()]
                = UrbanParMatId(static_cast<size_type>(UPMT::positron));
        }
        else
        {
            pid_to_pmdata[par_id.unchecked_get()]
                = UrbanParMatId(static_cast<size_type>(UPMT::muhad));
        }
    }
    make_builder(&host_data.pid_to_pmdata)
        .insert_back(pid_to_pmdata.begin(), pid_to_pmdata.end());

    // Coefficients for scaled Z: {electron, positron, muon/hadron}
    static Array<double, 3> const a_coeff{{0.87, 0.70, 0.87}};
    static Array<double, 3> const b_coeff{{2.0 / 3, 1.0 / 2, 2.0 / 3}};

    // Builders
    CollectionBuilder mdata(&host_data.material_data);
    CollectionBuilder pmdata(&host_data.par_mat_data);
    mdata.reserve(materials.num_materials());
    pmdata.reserve(host_data.num_par_mat * materials.num_materials());

    for (auto mat_id : range(MaterialId{materials.num_materials()}))
    {
        auto&& mat = materials.get(mat_id);

        // Build material-dependent data
        mdata.push_back(UrbanMscParams::calc_material_data(mat));

        // Build particle-dependent data
        double const zeff = mat.zeff();
        for (size_type p : range(host_data.num_par_mat))
        {
            UrbanMscParMatData this_pm;

            // Calculate scaled zeff
            this_pm.scaled_zeff = a_coeff[p] * fastpow(zeff, b_coeff[p]);

            // Compute the maximum distance that particles can travel
            // (different for electrons, hadrons)
            if (p != static_cast<size_type>(UPMT::muhad))
            {
                // Electrons and positrons
                this_pm.d_over_r = 9.6280e-1 - 8.4848e-2 * std::sqrt(zeff)
                                   + 4.3769e-3 * zeff;
                CELER_ASSERT(0 < this_pm.d_over_r && this_pm.d_over_r <= 1);
            }
            else
            {
                // Muons and charged hadrons
                this_pm.d_over_r = 1.15 - 9.76e-4 * zeff;
                CELER_ASSERT(0 < this_pm.d_over_r);
            }
            pmdata.push_back(this_pm);
        }
    }

    // Get the cross section energy grid limits (this checks that the limits
    // are the same for all particles/materials)
    auto energy_limit = helper.energy_grid_bounds();
    host_data.params.low_energy_limit = energy_limit[0];
    host_data.params.high_energy_limit = energy_limit[1];

    CELER_ASSERT(host_data);

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<UrbanMscData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build UrbanMsc data per material.
 *
 * Tabulated data based on G4UrbanMscModel::InitialiseModelCache() and
 * documented in section 8.1.5 of the Geant4 10.7 Physics Reference Manual.
 */
UrbanMscMaterialData
UrbanMscParams::calc_material_data(MaterialView const& material_view)
{
    using PolyQuad = PolyEvaluator<double, 2>;

    UrbanMscMaterialData data;

    double const zeff = material_view.zeff();

    // Linear+quadratic parameters for the step minimum calculation
    data.stepmin_coeff[0] = 1e3 * 27.725 / (1 + 0.203 * zeff);
    data.stepmin_coeff[1] = 1e3 * 6.152 / (1 + 0.111 * zeff);

    // Correction in the (modified Highland-Lynch-Dahl) theta_0 formula
    // (to be used in linear polynomial of log(E / MeV))
    double const z16 = fastpow(zeff, 1.0 / 6.0);
    double fz = PolyQuad(0.990395, -0.168386, 0.093286)(z16);
    data.theta_coeff[0] = fz * (1 - 8.7780e-2 / zeff);
    data.theta_coeff[1] = fz * (4.0780e-2 + 1.7315e-4 * zeff);

    // Tail parameters
    // (to be used in linear polynomial of tau^{1/6})
    double z13 = ipow<2>(z16);
    data.tail_coeff[0] = PolyQuad(2.3785, -4.1981e-1, 6.3100e-2)(z13);
    data.tail_coeff[1] = PolyQuad(4.7526e-1, 1.7694, -3.3885e-1)(z13);
    data.tail_coeff[2] = PolyQuad(2.3683e-1, -1.8111, 3.2774e-1)(z13);
    data.tail_corr = PolyQuad(1.7888e-2, 1.9659e-2, -2.6664e-3)(z13);

    CELER_ENSURE(data.theta_coeff[0] > 0 && data.theta_coeff[1] > 0);
    return data;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
