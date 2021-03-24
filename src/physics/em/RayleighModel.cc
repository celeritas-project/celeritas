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

#include "detail/RayleighData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
RayleighModel::RayleighModel(ModelId id, const ParticleParams& particles)
{
    CELER_EXPECT(id);

    HostValue host_pointers;

    host_pointers.model_id = id;
    host_pointers.gamma_id = particles.find(pdg::gamma());
    CELER_VALIDATE(host_pointers.model_id && host_pointers.gamma_id,
                   "gamma particles must be enabled to use the "
                   "Rayleigh Model.");

    this->build_data(&host_pointers);

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
 * Convert an Rayleigh data to a RayleighParameters and store.
 */
void RayleighModel::build_data(HostValue* pointers)
{
    auto data = make_builder(&pointers->params.data);
    data.reserve(detail::rayleigh_num_elements);

    Array<real_type, detail::rayleigh_num_parameters>
        parameter_arrays[detail::rayleigh_num_elements];

    for (auto i : range(detail::rayleigh_num_elements))
    {
        for (auto j : range(detail::rayleigh_num_parameters))
        {
            (parameter_arrays[i])[j] = detail::rayleigh_parameters[j][i];
        }
        data.push_back(parameter_arrays[i]);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
