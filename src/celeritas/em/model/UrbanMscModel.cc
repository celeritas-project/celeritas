//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/UrbanMscModel.cc
//---------------------------------------------------------------------------//
#include "UrbanMscModel.hh"

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
UrbanMscModel::UrbanMscModel(ActionId              id,
                             const ParticleParams& particles,
                             const MaterialParams& materials)
{
    CELER_EXPECT(id);
    HostValue host_ref;

    host_ref.ids.action   = id;
    host_ref.ids.electron = particles.find(pdg::electron());
    host_ref.ids.positron = particles.find(pdg::positron());
    CELER_VALIDATE(host_ref.ids.electron && host_ref.ids.positron,
                   << "missing e-/e+ (required for " << this->description()
                   << ")");

    // Save electron mass
    host_ref.electron_mass = particles.get(host_ref.ids.electron).mass();

    // Build UrbanMsc material data
    this->build_data(&host_ref, materials);

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<UrbanMscData>{std::move(host_ref)};

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
    electron_msc.lower    = zero_quantity();
    electron_msc.upper    = units::MevEnergy{1e+8};

    Applicability positron_msc = electron_msc;
    positron_msc.particle      = this->host_ref().ids.positron;

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
 * Construct UrbanMsc material data for all the materials in the problem.
 */
void UrbanMscModel::build_data(HostValue* data, const MaterialParams& materials)
{
    // Number of materials
    unsigned int num_materials = materials.num_materials();

    // Build msc data for available materials
    auto msc_data = make_builder(&data->msc_data);
    msc_data.reserve(num_materials);

    for (auto mat_id : range(MaterialId{num_materials}))
    {
        msc_data.push_back(
            UrbanMscModel::calc_material_data(materials.get(mat_id)));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build UrbanMsc data per material.
 *
 * Tabulated data based on G4UrbanMscModel::InitialiseModelCache() and
 * documented in section 8.1.5 of the Geant4 10.7 Physics Reference Manual.
 */
auto UrbanMscModel::calc_material_data(const MaterialView& material_view)
    -> MaterialData
{
    using PolyQuad = PolyEvaluator<double, 2>;

    if (CELER_UNLIKELY(material_view.num_elements() == 0))
    {
        // Pure vacuum, perhaps for testing?
        return {};
    }

    // Use double-precision for host-side setup since all the precomputed
    // factors are written as doubles
    double zeff{0};
    double norm{0};
    for (auto el_id : range(ElementComponentId{material_view.num_elements()}))
    {
        double weight = material_view.get_element_density(el_id);
        zeff += material_view.make_element_view(el_id).atomic_number() * weight;
        norm += weight;
    }
    zeff /= norm;
    CELER_ASSERT(zeff > 0);

    MaterialData data;

    data.zeff        = zeff;
    data.scaled_zeff = real_type(0.70) * std::sqrt(zeff);

    // Correction in the (modified Highland-Lynch-Dahl) theta_0 formula
    const double z16 = fastpow(zeff, 1.0 / 6.0);
    double       fz  = PolyQuad(0.990395, -0.168386, 0.093286)(z16);
    data.coeffth1    = fz * (1 - 8.7780e-2 / zeff);
    data.coeffth2    = fz * (4.0780e-2 + 1.7315e-4 * zeff);

    // Tail parameters
    double z13 = ipow<2>(z16);
    data.d[0]  = PolyQuad(2.3785, -4.1981e-1, 6.3100e-2)(z13);
    data.d[1]  = PolyQuad(4.7526e-1, 1.7694, -3.3885e-1)(z13);
    data.d[2]  = PolyQuad(2.3683e-1, -1.8111, 3.2774e-1)(z13);
    data.d[3]  = PolyQuad(1.7888e-2, 1.9659e-2, -2.6664e-3)(z13);

    data.z23 = ipow<2>(z13);

    // Parameters for the step minimum calculation
    data.stepmin_a = 1e3 * 27.725 / (1 + 0.203 * zeff);
    data.stepmin_b = 1e3 * 6.152 / (1 + 0.111 * zeff);

    // Parameters for the maximum distance that particles can travel
    data.d_over_r = 9.6280e-1 - 8.4848e-2 * std::sqrt(data.zeff)
                    + 4.3769e-3 * zeff;
    data.d_over_r_mh = 1.15 - 9.76e-4 * zeff;

    return data;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
