//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionMirror.hh"
#include "physics/base/Model.hh"
#include "physics/material/MaterialParams.hh"
#include "detail/Rayleigh.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch Rayleigh scattering.
 */
class RayleighModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using HostRef
        = detail::RayleighGroup<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = detail::RayleighGroup<Ownership::const_reference, MemSpace::device>;
    //@}

  public:
    // Construct from model ID and other necessary data
    RayleighModel(ModelId               id,
                  const ParticleParams& particles,
                  const MaterialParams& materials);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const ModelInteractPointers&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Rayleigh Scattering"; }

    //! Access Rayleigh pointers on the host
    const HostRef& host_group() const { return group_.host(); }

    //! Access Rayleigh pointers on the device
    const DeviceRef& device_group() const { return group_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<detail::RayleighGroup> group_;

  private:
    using HostValue = detail::RayleighGroup<Ownership::value, MemSpace::host>;
    void build_data(HostValue* group, const MaterialParams& materials);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
