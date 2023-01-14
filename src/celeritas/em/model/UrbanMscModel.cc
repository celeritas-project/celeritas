//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/UrbanMscModel.cc
//---------------------------------------------------------------------------//
#include "UrbanMscModel.hh"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridInserter.hh"
#include "celeritas/grid/XsGridData.hh"
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
 * Construct from model ID and other necessary data.
 */
UrbanMscModel::UrbanMscModel(ActionId id,
                             ParticleParams const& particles,
                             MaterialParams const& materials,
                             ImportedProcessAdapter const& pdata)
{
    CELER_EXPECT(id);
    HostValue host_data;

    host_data.ids.action = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());
    CELER_VALIDATE(host_data.ids.electron && host_data.ids.positron,
                   << "missing e-/e+ (required for " << this->description()
                   << ")");

    // Save electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    {
        // Particle-dependent data
        Array<ParticleId, 2> const par_ids{
            {host_data.ids.electron, host_data.ids.positron}};
        Array<ImportPhysicsTable const*, 2> const xs_tables{{
            &pdata.get_lambda(par_ids[0]),
            &pdata.get_lambda(par_ids[1]),
        }};
        CELER_ASSERT(xs_tables[0]->x_units == ImportUnits::mev);
        CELER_ASSERT(xs_tables[0]->y_units == ImportUnits::cm_inv);
        CELER_ASSERT(xs_tables[1]->x_units == ImportUnits::mev);
        CELER_ASSERT(xs_tables[1]->y_units == ImportUnits::cm_inv);

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
            mdata.push_back(UrbanMscModel::calc_material_data(mat));

            // Build particle-dependent data
            const real_type zeff = mat.zeff();
            for (size_type p : range(par_ids.size()))
            {
                UrbanMscParMatData this_pm;

                // Calculate scaled zeff
                this_pm.scaled_zeff = a_coeff[p] * fastpow(zeff, b_coeff[p]);

                // Get the cross section data for this particle and material
                ImportPhysicsVector const& pvec
                    = xs_tables[p]->physics_vectors[mat_id.unchecked_get()];
                CELER_ASSERT(pvec.vector_type == ImportPhysicsVectorType::log);

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
 * Particle types and energy ranges that this model applies to.
 */
auto UrbanMscModel::applicability() const -> SetApplicability
{
    Applicability electron_msc;
    electron_msc.particle = this->host_ref().ids.electron;
    electron_msc.lower = zero_quantity();
    electron_msc.upper = units::MevEnergy{1e+8};

    Applicability positron_msc = electron_msc;
    positron_msc.particle = this->host_ref().ids.positron;

    return {electron_msc, positron_msc};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto UrbanMscModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // No cross sections for multiple scattering
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * No discrete interaction: it's integrated into along_step.
 */
void UrbanMscModel::execute(CoreDeviceRef const&) const {}
void UrbanMscModel::execute(CoreHostRef const&) const {}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId UrbanMscModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Build UrbanMsc data per material.
 *
 * Tabulated data based on G4UrbanMscModel::InitialiseModelCache() and
 * documented in section 8.1.5 of the Geant4 10.7 Physics Reference Manual.
 */
UrbanMscMaterialData
UrbanMscModel::calc_material_data(MaterialView const& material_view)
{
    using PolyQuad = PolyEvaluator<double, 2>;

    MaterialData data;

    double zeff = material_view.zeff();

    // Correction in the (modified Highland-Lynch-Dahl) theta_0 formula
    double const z16 = fastpow(zeff, 1.0 / 6.0);
    double fz = PolyQuad(0.990395, -0.168386, 0.093286)(z16);
    data.coeffth1 = fz * (1 - 8.7780e-2 / zeff);
    data.coeffth2 = fz * (4.0780e-2 + 1.7315e-4 * zeff);

    // Tail parameters
    double z13 = ipow<2>(z16);
    data.d[0] = PolyQuad(2.3785, -4.1981e-1, 6.3100e-2)(z13);
    data.d[1] = PolyQuad(4.7526e-1, 1.7694, -3.3885e-1)(z13);
    data.d[2] = PolyQuad(2.3683e-1, -1.8111, 3.2774e-1)(z13);
    data.d[3] = PolyQuad(1.7888e-2, 1.9659e-2, -2.6664e-3)(z13);

    // Parameters for the step minimum calculation
    data.stepmin_a = 1e3 * 27.725 / (1 + 0.203 * zeff);
    data.stepmin_b = 1e3 * 6.152 / (1 + 0.111 * zeff);

    // Parameters for the maximum distance that particles can travel
    data.d_over_r = 9.6280e-1 - 8.4848e-2 * std::sqrt(zeff) + 4.3769e-3 * zeff;
    data.d_over_r_mh = 1.15 - 9.76e-4 * zeff;

    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
