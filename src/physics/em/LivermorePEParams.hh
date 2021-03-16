//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "io/ImportPhysicsVector.hh"
#include "physics/material/Types.hh"
#include "LivermorePEInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for the Livermore EPICS2014 Electron Photon Interaction
 * Cross Section library.
 *
 * \todo Combine with livermore model and move the input description to `io/`.
 * \todo Use Collection and CollectionMirror for data storage.
 */
class LivermorePEParams
{
  public:
    //@{
    //! Type aliases
    //@}

    struct SubshellInput
    {
        using EnergyUnits = units::Mev;
        using XsUnits     = units::Barn;
        using Energy      = Quantity<EnergyUnits>;

        Energy                 binding_energy; //!< Ionization energy
        std::vector<real_type> param_low;  //!< Low energy xs fit parameters
        std::vector<real_type> param_high; //!< High energy xs fit parameters
        std::vector<real_type> xs;         //!< Tabulated cross sections
        std::vector<real_type> energy;     //!< Tabulated energies
    };

    struct ElementInput
    {
        using Energy = SubshellInput::Energy;

        ImportPhysicsVector xs_low;      //!< Low energy range tabulated xs
        ImportPhysicsVector xs_high;     //!< High energy range tabulated xs
        Energy              thresh_low;  //!< Threshold for low energy fit
        Energy              thresh_high; //!< Threshold for high energy fit
        std::vector<SubshellInput> shells;
    };

    struct Input
    {
        std::vector<ElementInput> elements;
    };

  public:
    // Construct with a vector of element identifiers
    explicit LivermorePEParams(const Input& inp);

    // Access Livermore data on the host
    LivermorePEParamsPointers host_pointers() const;

    // Access Livermore data on the device
    LivermorePEParamsPointers device_pointers() const;

  private:
    std::vector<LivermoreElement>  host_elements_;
    std::vector<LivermoreSubshell> host_shells_;
    std::vector<real_type>         host_data_;

    DeviceVector<LivermoreElement>  device_elements_;
    DeviceVector<LivermoreSubshell> device_shells_;
    DeviceVector<real_type>         device_data_;

    // HELPER FUNCTIONS
    void                    append_livermore_element(const ElementInput& inp);
    Span<LivermoreSubshell> extend_shells(const ElementInput& inp);
    Span<real_type>         extend_data(const std::vector<real_type>& data);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
