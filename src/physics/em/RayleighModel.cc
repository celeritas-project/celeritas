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

    HostValue host_pointers;

    host_pointers.model_id = id;
    host_pointers.gamma_id = particles.find(pdg::gamma());
    CELER_VALIDATE(host_pointers.model_id && host_pointers.gamma_id,
                   "gamma particles must be enabled to use the "
                   "Rayleigh Model.");

    this->build_data(&host_pointers, materials);

    // Move to mirrored data, copying to device
    pointers_
        = CollectionMirror<detail::RayleighPointers>{std::move(host_pointers)};

    CELER_ENSURE(this->pointers_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto RayleighModel::applicability() const -> SetApplicability
{
    Applicability rayleigh_scattering;
    rayleigh_scattering.particle = this->host_pointers().gamma_id;
    rayleigh_scattering.lower    = units::MevEnergy{1e-5};
    rayleigh_scattering.upper    = units::MevEnergy{1e+8};

    return {rayleigh_scattering};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void RayleighModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractPointers& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::rayleigh_interact(this->device_pointers(), pointers);
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
    return this->host_pointers().model_id;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an RayleighData to a RayleighParameters and store.
 */
void RayleighModel::build_data(HostValue*            pointers,
                               const MaterialParams& materials)
{
    // Number of elements
    unsigned int num_elements = materials.num_elements();

    // Build data for available elements
    using RayleighData = detail::RayleighData;

    auto data_n = make_builder(&pointers->params.data_n);
    auto data_b = make_builder(&pointers->params.data_b);
    auto data_x = make_builder(&pointers->params.data_x);

    data_n.reserve(num_elements);
    data_b.reserve(num_elements);
    data_x.reserve(num_elements);

    for (auto el_id : range(ElementId{num_elements}))
    {
        unsigned int z = materials.get(el_id).atomic_number() - 1;
        CELER_ASSERT(z < detail::rayleigh_num_elements);

        Real3 n_array;
        Real3 b_array;
        Real3 x_array;

        for (auto j : range(3))
        {
            n_array[j] = RayleighData::angular_parameters[j + 6][z] - 1.0;
            b_array[j] = RayleighData::angular_parameters[j + 3][z];
            x_array[j] = RayleighData::angular_parameters[j][z];
        }
        data_n.push_back(n_array);
        data_b.push_back(b_array);
        data_x.push_back(x_array);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
