//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/LivermorePEModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
struct ImportLivermorePE;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Livermore photoelectric model interaction.
 */
class LivermorePEModel final : public Model
{
  public:
    //!@{
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<ImportLivermorePE(AtomicNumber)>;
    using HostRef = LivermorePEHostRef;
    using DeviceRef = LivermorePEDeviceRef;
    //!@}

  public:
    // Construct from model ID and other necessary data
    LivermorePEModel(ActionId id,
                     ParticleParams const& particles,
                     MaterialParams const& materials,
                     ReadData load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "photoel-livermore"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by Livermore photoelectric effect";
    }

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<LivermorePEData> data_;

    using HostXsData = HostVal<LivermorePEXsData>;
    void
    append_element(ImportLivermorePE const& inp, HostXsData* xs_data) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
