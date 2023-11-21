//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMscParams.cc
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
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridInserter.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/PDGNumber.hh"
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
                            ImportData const& import)
{
    auto is_urban = [](ImportMscModel const& imm) {
        return imm.model_class == ImportModelClass::urban_msc;
    };
    if (!std::any_of(
            import.msc_models.begin(), import.msc_models.end(), is_urban))
    {
        // No Urban MSC present
        return nullptr;
    }

    Options opts;
    opts.lambda_limit = import.em_params.msc_lambda_limit;
    opts.safety_fact = import.em_params.msc_safety_factor;
    opts.range_fact = import.em_params.msc_range_factor;

    return std::make_shared<UrbanMscParams>(
        particles, materials, import.msc_models, opts);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and material properties.
 */
UrbanMscParams::UrbanMscParams(ParticleParams const& particles,
                               MaterialParams const& materials,
                               VecImportMscModel const& mdata_vec,
                               Options options)
{
    using units::MevEnergy;

    ScopedMem record_mem("UrbanMscParams.construct");

    HostVal<UrbanMscData> host_data;

    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());
    CELER_VALIDATE(host_data.ids.electron && host_data.ids.positron,
                   << "missing e-/e+ (required for Urban MSC)");

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // TODO: change IDs to a vector for all particles. This model should apply
    // to muons and charged hadrons as well
    if (particles.find(pdg::mu_minus()) || particles.find(pdg::mu_plus())
        || particles.find(pdg::proton()))
    {
        CELER_LOG(warning) << "Multiple scattering is not implemented for for "
                              "particles other than electron and positron";
    }

    // Save parameters
    CELER_VALIDATE(options.lambda_limit > 0,
                   << "invalid lambda_limit=" << options.lambda_limit
                   << " (should be positive)");
    CELER_VALIDATE(options.safety_fact >= 0.1,
                   << "invalid safety_fact=" << options.safety_fact
                   << " (should be >= 0.1)");
    CELER_VALIDATE(options.range_fact > 0 && options.range_fact < 1,
                   << "invalid range_fact=" << options.range_fact
                   << " (should be within 0 < limit < 1)");
    host_data.params.lambda_limit = options.lambda_limit;
    host_data.params.range_fact = options.range_fact;
    host_data.params.safety_fact = options.safety_fact;

    // Filter MSC data by model and particle type
    std::vector<ImportMscModel const*> urban_data(particles.size(), nullptr);
    for (ImportMscModel const& imm : mdata_vec)
    {
        // Filter out other MSC models
        if (imm.model_class != ImportModelClass::urban_msc)
            continue;

        // Filter out unused particles
        PDGNumber pdg{imm.particle_pdg};
        ParticleId pid = pdg ? particles.find(pdg) : ParticleId{};
        if (!pid)
            continue;

        if (!urban_data[pid.get()])
        {
            // Save data
            urban_data[pid.get()] = &imm;
        }
        else
        {
            // Warn: possibly multiple physics lists or different models in
            // different regions
            CELER_LOG(warning)
                << "duplicate " << to_cstring(imm.model_class)
                << " physics data for particle " << particles.id_to_label(pid)
                << ": ignoring all but the first encountered model";
        }
    }

    auto get_scaled_xs = [&urban_data, &particles](ParticleId pid) {
        CELER_ASSERT(pid < urban_data.size());
        ImportMscModel const* imm = urban_data[pid.unchecked_get()];
        CELER_VALIDATE(imm,
                       << "missing Urban MSC physics data for particle "
                       << particles.id_to_label(pid));
        return &imm->xs_table;
    };

    {
        // Set initial high/low energy limits
        auto const& phys_vec
            = get_scaled_xs(host_data.ids.electron)->physics_vectors;
        CELER_ASSERT(!phys_vec.empty());
        CELER_ASSERT(!phys_vec[0].x.empty());
        host_data.params.low_energy_limit = MevEnergy(phys_vec[0].x.front());
        host_data.params.high_energy_limit = MevEnergy(phys_vec[0].x.back());
    }

    {
        // Particle-dependent data
        Array<ParticleId, 2> const par_ids{
            {host_data.ids.electron, host_data.ids.positron}};
        Array<ImportPhysicsTable const*, 2> const xs_tables{{
            get_scaled_xs(par_ids[0]),
            get_scaled_xs(par_ids[1]),
        }};
        CELER_ASSERT(xs_tables[0]->x_units == ImportUnits::mev);
        CELER_ASSERT(xs_tables[0]->y_units == ImportUnits::mev_2_per_cm);
        CELER_ASSERT(xs_tables[1]->x_units == ImportUnits::mev);
        CELER_ASSERT(xs_tables[1]->y_units == ImportUnits::mev_2_per_cm);

        // Coefficients for scaled Z
        static Array<double, 2> const a_coeff{{0.87, 0.70}};
        static Array<double, 2> const b_coeff{{2.0 / 3, 1.0 / 2}};

        // Builders
        auto mdata = make_builder(&host_data.material_data);
        auto pmdata = make_builder(&host_data.par_mat_data);
        mdata.reserve(materials.num_materials());
        pmdata.reserve(2 * materials.num_materials());

        // TODO: simplify when refactoring ValueGridInserter, etc
        ValueGridInserter::XsGridCollection xgc;
        ValueGridInserter vgi{&host_data.reals, &xgc};

        for (auto mat_id : range(MaterialId{materials.num_materials()}))
        {
            auto&& mat = materials.get(mat_id);

            // Build material-dependent data
            mdata.push_back(UrbanMscParams::calc_material_data(mat));

            // Build particle-dependent data
            double const zeff = mat.zeff();
            for (size_type p : range(par_ids.size()))
            {
                UrbanMscParMatData this_pm;

                // Calculate scaled zeff
                this_pm.scaled_zeff = a_coeff[p] * fastpow(zeff, b_coeff[p]);

                // Compute the maximum distance that particles can travel
                // (different for electrons, hadrons)
                if (par_ids[p] == host_data.ids.electron
                    || par_ids[p] == host_data.ids.positron)
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

                // Get the cross section data for this particle and material
                ImportPhysicsVector const& pvec
                    = xs_tables[p]->physics_vectors[mat_id.unchecked_get()];
                CELER_ASSERT(pvec.vector_type == ImportPhysicsVectorType::log);

                // Check that the limits are the same for all materials and
                // particles; otherwise we need to change
                // `UrbanMsc::is_applicable` to look up the particle and
                // material
                CELER_VALIDATE(host_data.params.low_energy_limit
                                       == MevEnergy(pvec.x.front())
                                   && host_data.params.high_energy_limit
                                          == MevEnergy(pvec.x.back()),
                               << "multiple scattering cross section energy "
                                  "limits are inconsistent across particles "
                                  "and/or materials");

                // To reuse existing code (TODO: simplify when refactoring)
                // use the value grid builder to construct the grid entry in a
                // temporary container and then copy it into the pm data.
                auto vgb = ValueGridLogBuilder::from_geant(make_span(pvec.x),
                                                           make_span(pvec.y));
                auto grid_id = vgb->build(vgi);
                CELER_ASSERT(grid_id.get() == pmdata.size());
                this_pm.xs = xgc[grid_id];
                pmdata.push_back(this_pm);
                CELER_ASSERT(host_data.at(mat_id, par_ids[p]).get() + 1
                             == host_data.par_mat_data.size());
            }
        }
        CELER_ASSERT(host_data);
    }

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<UrbanMscData>{std::move(host_data)};

    CELER_ENSURE(this->mirror_);
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
