//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/UrbanMscModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;
class MaterialParams;
class MaterialView;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch UrbanMsc model.
 */
class UrbanMscModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using HostRef = UrbanMscData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = UrbanMscData<Ownership::const_reference, MemSpace::device>;
    //@}

  public:
    // Construct from model ID and other necessary data
    UrbanMscModel(ActionId              id,
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
    std::string label() const final { return "msc-urban"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "Urban multiple scattering";
    }

    //! Access UrbanMsc data on the host
    const HostRef& host_ref() const { return mirror_.host(); }

    //! Access UrbanMsc data on the device
    const DeviceRef& device_ref() const { return mirror_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<UrbanMscData> mirror_;

    //// TYPES ////

    using HostValue    = UrbanMscData<Ownership::value, MemSpace::host>;
    using MaterialData = UrbanMscMaterialData;

    //// HELPER FUNCTIONS ////

    void build_data(HostValue* host_data, const MaterialParams& materials);
    MaterialData calc_material_data(const MaterialView& material_view);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
