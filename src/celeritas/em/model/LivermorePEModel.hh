//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/LivermorePEModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/io/ImportLivermorePE.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Livermore photoelectric model interaction.
 */
class LivermorePEModel final : public Model
{
  public:
    //!@{
    using AtomicNumber = int;
    using MevEnergy    = units::MevEnergy;
    using ReadData     = std::function<ImportLivermorePE(AtomicNumber)>;
    using HostRef      = detail::LivermorePEHostRef;
    using DeviceRef    = detail::LivermorePEDeviceRef;
    //!@}

  public:
    // Construct from model ID and other necessary data
    LivermorePEModel(ActionId              id,
                     const ParticleParams& particles,
                     const MaterialParams& materials,
                     ReadData              load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel on device
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "photoel-livermore"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "Livermore photoelectric effect";
    }

    //! Access data on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access data on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<detail::LivermorePEData> data_;

    using HostXsData
        = detail::LivermorePEXsData<Ownership::value, MemSpace::host>;
    void
    append_element(const ImportLivermorePE& inp, HostXsData* xs_data) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
