//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/CollectionBuilder.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
RayleighModel::RayleighModel(ModelId               id,
                             const ParticleParams& particles,
                             const MaterialParams& materials)
{
    CELER_EXPECT(id);

    HostValue host_group;

    host_group.model_id = id;
    host_group.gamma_id = particles.find(pdg::gamma());
    CELER_VALIDATE(host_group.model_id && host_group.gamma_id,
                   "gamma particles must be enabled to use the "
                   "Rayleigh Model.");

    this->build_data(&host_group, materials);

    // Move to mirrored data, copying to device
    group_
        = CollectionMirror<detail::RayleighGroup>{std::move(host_group)};

    CELER_ENSURE(this->group_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto RayleighModel::applicability() const -> SetApplicability
{
    Applicability rayleigh_scattering;
    rayleigh_scattering.particle = this->host_group().gamma_id;
    rayleigh_scattering.lower    = units::MevEnergy{1e-5};
    rayleigh_scattering.upper    = units::MevEnergy{1e+8};

    return {rayleigh_scattering};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void RayleighModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractPointers& group) const
{
#if CELERITAS_USE_CUDA
    detail::rayleigh_interact(this->device_group(), group);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId RayleighModel::model_id() const
{
    return this->host_group().model_id;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an RayleighData to a RayleighParameters and store.
 */
void RayleighModel::build_data(HostValue*            group,
                               const MaterialParams& materials)
{
    // Number of elements
    unsigned int num_elements = materials.num_elements();

    // Build data for available elements
    using RayleighData = detail::RayleighData;

    auto data = make_builder(&group->params.data);
    data.reserve(num_elements);

    for (auto el_id : range(ElementId{num_elements}))
    {
        unsigned int z = materials.get(el_id).atomic_number() - 1;
        CELER_ASSERT(z < RayleighData::num_elements);

        detail::RayleighElementData el_data;

        for (auto j : range(3))
        {
            el_data.a[j] = RayleighData::angular_parameters[j][z];
            el_data.b[j] = RayleighData::angular_parameters[j + 3][z];
            el_data.n[j] = RayleighData::angular_parameters[j + 6][z] - 1.0;
        }
        data.push_back(el_data);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
