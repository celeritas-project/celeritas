//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMscModel.cc
//---------------------------------------------------------------------------//
#include "UrbanMscModel.hh"

#include "base/Algorithms.hh"
#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/CollectionBuilder.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/em/detail/UrbanMscData.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/material/MaterialView.hh"

#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
UrbanMscModel::UrbanMscModel(ModelId               id,
                             const ParticleParams& particles,
                             const MaterialParams& materials)
{
    CELER_EXPECT(id);
    HostValue host_ref;

    host_ref.model_id    = id;
    host_ref.electron_id = particles.find(pdg::electron());
    host_ref.positron_id = particles.find(pdg::positron());
    CELER_VALIDATE(host_ref.electron_id && host_ref.positron_id,
                   << "missing e-/e+ (required for " << this->label() << ")");

    // Save electron mass
    host_ref.electron_mass = particles.get(host_ref.electron_id).mass();

    // Build UrbanMsc material data
    this->build_data(&host_ref, materials);

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<detail::UrbanMscData>{std::move(host_ref)};

    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto UrbanMscModel::applicability() const -> SetApplicability
{
    Applicability electron_msc;
    electron_msc.particle = this->host_ref().electron_id;
    electron_msc.lower    = zero_quantity();
    electron_msc.upper    = units::MevEnergy{1e+8};

    Applicability positron_msc = electron_msc;
    positron_msc.particle      = this->host_ref().positron_id;

    return {electron_msc, positron_msc};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void UrbanMscModel::interact(const DeviceInteractRef&) const
{
    CELER_NOT_IMPLEMENTED("host interactions: do not apply for msc");
}

void UrbanMscModel::interact(const HostInteractRef&) const
{
    CELER_NOT_IMPLEMENTED("device interactions: do not apply for msc");
}
//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId UrbanMscModel::model_id() const
{
    return this->host_ref().model_id;
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
 * Tabulated data based on G4UrbanMscModel::InitialiseModelCache()
 */
auto UrbanMscModel::calc_material_data(const MaterialView& material_view)
    -> MaterialData
{
    real_type                     zeff{0};
    real_type                     norm{0};
    ElementComponentId::size_type num_elements = material_view.num_elements();
    for (auto el_id : range(ElementComponentId{num_elements}))
    {
        real_type weight = material_view.get_element_density(el_id);
        zeff += material_view.element_view(el_id).atomic_number() * weight;
        norm += weight;
    }
    zeff /= norm;

    CELER_EXPECT(zeff > 0);
    MaterialData data;

    data.zeff       = zeff;
    real_type log_z = std::log(zeff);

    // Correction in the (modified Highland-Lynch-Dahl) theta_0 formula
    real_type w   = std::exp(log_z / 6);
    real_type fz  = 0.990395 + w * (-0.168386 + w * 0.093286);
    data.coeffth1 = fz * (1 - 8.7780e-2 / zeff);
    data.coeffth2 = fz * (4.0780e-2 + 1.7315e-4 * zeff);

    // Tail parameters
    real_type z13 = ipow<2>(w);
    data.coeffc1  = 2.3785 - z13 * (4.1981e-1 - z13 * 6.3100e-2);
    data.coeffc2  = 4.7526e-1 + z13 * (1.7694 - z13 * 3.3885e-1);
    data.coeffc3  = 2.3683e-1 - z13 * (1.8111 - z13 * 3.2774e-1);
    data.coeffc4  = 1.7888e-2 + z13 * (1.9659e-2 - z13 * 2.6664e-3);

    data.z23 = ipow<2>(z13);

    data.stepmina = 27.725 / (1 + 0.203 * zeff);
    data.stepminb = 6.152 / (1 + 0.111 * zeff);

    data.doverra = 9.6280e-1 - 8.4848e-2 * std::sqrt(data.zeff)
                   + 4.3769e-3 * zeff;
    data.doverrb = 1.15 - 9.76e-4 * zeff;

    return data;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
