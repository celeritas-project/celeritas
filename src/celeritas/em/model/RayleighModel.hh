//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/RayleighData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Model.hh"

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
    using HostRef = RayleighData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = RayleighData<Ownership::const_reference, MemSpace::device>;
    //@}

  public:
    // Construct from model ID and other necessary data
    RayleighModel(ActionId              id,
                  const ParticleParams& particles,
                  const MaterialParams& materials);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel to host data
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "scat-rayleigh"; }

    //! Name of the model, for user interaction
    std::string description() const final { return "Rayleigh scattering"; }

    //! Access Rayleigh data on the host
    const HostRef& host_ref() const { return mirror_.host(); }

    //! Access Rayleigh data on the device
    const DeviceRef& device_ref() const { return mirror_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<RayleighData> mirror_;

    //// TYPES ////

    using AtomicNumber = int;
    using HostValue    = RayleighData<Ownership::value, MemSpace::host>;
    using ElScatParams = RayleighParameters;

    //// HELPER FUNCTIONS ////

    void build_data(HostValue* host_data, const MaterialParams& materials);
    static const ElScatParams& get_el_parameters(AtomicNumber atomic_number);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
