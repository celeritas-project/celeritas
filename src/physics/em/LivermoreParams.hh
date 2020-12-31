//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "io/ImportPhysicsVector.hh"
#include "physics/material/Types.hh"
#include "LivermoreParamsPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for the Livermore EPICS2014 Electron Photon Interaction
 * Cross Section library.
 */
class LivermoreParams
{
  public:
    //@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //@}

    struct SubshellInput
    {
        units::MevEnergy       binding_energy; //!< Ionization energy
        std::vector<real_type> param_low;  //!< Low energy xs fit parameters
        std::vector<real_type> param_high; //!< High energy xs fit parameters
        std::vector<real_type> xs;         //!< Tabulated cross sections
        std::vector<real_type> energy;     //!< Tabulated energies
    };

    struct ElementInput
    {
        ElementDefId        el_id;      //!< Index in MaterialParams elements
        ImportPhysicsVector xs_low;     //!< Low energy range tabulated xs
        ImportPhysicsVector xs_high;    //!< High energy range tabulated xs
        units::MevEnergy    thresh_low; //!< Threshold for using low energy fit
        units::MevEnergy thresh_high; //!< Threshold for using high energy fit
        std::vector<SubshellInput> shells;
    };

    struct Input
    {
        std::vector<ElementInput> elements;
    };

  public:
    // Construct with a vector of element identifiers
    explicit LivermoreParams(Input inp);

    // Access Livermore data on the host
    LivermoreParamsPointers host_pointers() const;

    // Access Livermore data on the device
    LivermoreParamsPointers device_pointers() const;

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
