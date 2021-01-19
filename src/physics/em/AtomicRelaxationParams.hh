//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>
#include "base/DeviceVector.hh"
#include "AtomicRelaxationInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for the EADL transition data for atomic relaxation.
 */
class AtomicRelaxationParams
{
  public:
    //@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;

    //@}

    //! Radiative transition input
    struct FluorSubshellInput
    {
        std::vector<size_type> initial_shell; //!< Originating shell designator
        std::vector<real_type> transition_energy;
        std::vector<real_type> transition_prob;
    };

    //! Non-radiative transition input
    struct AugerSubshellInput
    {
        std::vector<size_type> initial_shell; //!< Originating shell designator
        std::vector<size_type> auger_shell;   //!< Auger shell designator
        std::vector<real_type> transition_energy;
        std::vector<real_type> transition_prob;
    };

    struct ElementInput
    {
        using EnergyUnits = units::Mev;

        std::vector<size_type> designators;    //!< EADL subshell designator
        std::vector<FluorSubshellInput> fluor; //!< Radiative transitions
        std::vector<AugerSubshellInput> auger; //!< Non-radiative transistions
    };

    struct Input
    {
        std::vector<ElementInput> elements;
        bool is_auger_enabled{false}; //!< Whether to produce Auger electrons
        ParticleId electron_id{};
        ParticleId gamma_id{};
    };

  public:
    // Construct with a vector of element identifiers
    explicit AtomicRelaxationParams(const Input& inp);

    // Access EADL data on the host
    AtomicRelaxParamsPointers host_pointers() const;

    // Access EADL data on the device
    AtomicRelaxParamsPointers device_pointers() const;

    //! Flag a subshell index as unassigned
    static CELER_CONSTEXPR_FUNCTION size_type unassigned()
    {
        return size_type(-1);
    }

  private:
    //// HOST DATA ////

    bool                             is_auger_enabled_;
    ParticleId                       electron_id_;
    ParticleId                       gamma_id_;
    std::vector<AtomicRelaxElement>  host_elements_;
    std::vector<AtomicRelaxSubshell> host_shells_;
    std::vector<size_type>           host_id_data_;
    std::vector<real_type>           host_tr_data_;

    //// DEVICE DATA ////

    DeviceVector<AtomicRelaxElement>  device_elements_;
    DeviceVector<AtomicRelaxSubshell> device_shells_;
    DeviceVector<size_type>           device_id_data_;
    DeviceVector<real_type>           device_tr_data_;

    // HELPER FUNCTIONS
    void                      append_element(const ElementInput& inp);
    Span<AtomicRelaxSubshell> extend_shells(const ElementInput& inp);
    void map_to_index(const std::unordered_map<size_type, size_type>& des_to_id,
                      const std::vector<size_type>&                   des,
                      std::vector<size_type>*                         id);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
