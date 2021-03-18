//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"

#include <functional>
#include "base/CollectionMirror.hh"
#include "io/ImportSBTable.hh"
#include "detail/SeltzerBerger.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Manage the Seltzer-Berger model for Bremsstrahlung.
 *
 * The total scaled bremsstrahlung differential cross section for an element Z
 * is defined as
 * \f[
 *   \chi(Z,E,\kappa) = \frac{\beta^2}{Z^2} k \frac{d \sigma}{dk},
 * \f]
 * where \f$ \kappa = k / E \f$ is the ratio of the emitted photon energy to
 * the incident charged particle energy, \f$ \beta \f$ is the ratio of the
 * charged particle velocity to the speed of light, and \f$ \frac{d \sigma}{dk}
 * \f$ is the bremsstrahlung differential cross section.
 *
 * Seltzer and Berger have tabulated the scaled DCS (in mb) for elements Z = 1
 * - 100 and for incident charged particle energies from 1 keV to 10 GeV
 * (reported in MeV) in Seltzer S.M. and M.J. Berger (1986), "Bremsstrahlung
 * energy spectra from electrons with kinetic energy 1 keV–10 GeV incident on
 * screened nuclei and orbital electrons of neutral atoms with Z = 1–100", At.
 * Data Nucl. Data Tables 35, 345–418.
 */
class SeltzerBergerModel final : public Model
{
  public:
    //!@{
    using AtomicNumber = int;
    using ReadData     = std::function<ImportSBTable(AtomicNumber)>;
    using HostRef
        = detail::SeltzerBergerData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = detail::SeltzerBergerData<Ownership::const_reference, MemSpace::device>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    SeltzerBergerModel(ModelId               id,
                       const ParticleParams& particles,
                       const MaterialParams& materials,
                       ReadData              load_sb_table);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const ModelInteractPointers&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Seltzer-Berger"; }

    //! Access SB data on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access SB data on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<detail::SeltzerBergerData> data_;

    using HostXsTables
        = detail::SeltzerBergerTableData<Ownership::value, MemSpace::host>;
    void append_table(const ImportSBTable& table, HostXsTables* tables) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
