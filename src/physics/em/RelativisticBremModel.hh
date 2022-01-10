//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"

#include "base/CollectionMirror.hh"
#include "physics/material/MaterialParams.hh"
#include "detail/RelativisticBremData.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the relativistic Bremsstrahlung model for high-energy
 * electrons and positrons with the Landau-Pomeranchuk-Migdal (LPM) effect
 */
class RelativisticBremModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using HostRef   = detail::RelativisticBremData<Ownership::const_reference,
                                                 MemSpace::host>;
    using DeviceRef = detail::RelativisticBremData<Ownership::const_reference,
                                                   MemSpace::device>;
    //@}

  public:
    // Construct from model ID and other necessary data
    RelativisticBremModel(ModelId               id,
                          const ParticleParams& particles,
                          const MaterialParams& materials,
                          bool                  enable_lpm = true);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel to host data
    void interact(const HostInteractRef&) const final;

    // Apply the interaction kernel to device data
    void interact(const DeviceInteractRef&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Relativistic Bremsstrahlung"; }

    //! Access data on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access data on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<detail::RelativisticBremData> data_;

    //// TYPES ////

    using HostValue
        = detail::RelativisticBremData<Ownership::value, MemSpace::host>;

    using AtomicNumber = int;
    using FormFactor   = detail::RelBremFormFactor;
    using MigdalData   = detail::RelBremMigdalData;
    using ElementData  = detail::RelBremElementData;

    //// HELPER FUNCTIONS ////

    void build_data(HostValue*            host_data,
                    const MaterialParams& materials,
                    real_type             particle_mass);

    static const FormFactor& get_form_factor(AtomicNumber index);
    MigdalData               compute_lpm_data(real_type shat);
    ElementData
    compute_element_data(const ElementView& elem, real_type particle_mass);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
