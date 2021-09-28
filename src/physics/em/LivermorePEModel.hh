//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"

#include <functional>
#include "base/CollectionMirror.hh"
#include "io/ImportLivermorePE.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "detail/LivermorePEInteractor.hh"

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
    LivermorePEModel(ModelId               id,
                     const ParticleParams& particles,
                     const MaterialParams& materials,
                     ReadData              load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void interact(const HostInteractRefs&) const final;

    // Apply the interaction kernel on device
    void interact(const DeviceInteractRefs&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Livermore photoelectric"; }

    //! Access data on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access data on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

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
