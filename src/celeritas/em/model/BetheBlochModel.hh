//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheBlochModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/BetheBlochData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Bethe-Bloch ionization model interaction.
 */
class BetheBlochModel final : public Model, public StaticConcreteAction
{
  public:
    //@{
    //! Type aliases
    using HostRef = HostCRef<BetheBlochData>;
    using DeviceRef = DeviceCRef<BetheBlochData>;
    //@}

  public:
    // Construct from model ID and other necessary data
    BetheBlochModel(ActionId, ParticleParams const&, SetApplicability);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    // Host/device storage and reference
    CollectionMirror<BetheBlochData> data_;
    // Particle types and energy ranges that this model applies to
    SetApplicability applicability_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
