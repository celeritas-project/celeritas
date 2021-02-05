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

    struct TransitionInput
    {
        size_type initial_shell; //!< Originating shell designator
        size_type auger_shell;   //!< Auger shell designator
        real_type probability;
        real_type energy;
    };

    struct SubshellInput
    {
        size_type                    designator;
        std::vector<TransitionInput> fluor;
        std::vector<TransitionInput> auger;
    };

    struct ElementInput
    {
        std::vector<SubshellInput> shells;
    };

    struct Input
    {
        using EnergyUnits = units::Mev;

        std::vector<ElementInput> elements;
        ParticleId                electron_id{};
        ParticleId                gamma_id{};
        bool is_auger_enabled{false}; //!< Whether to produce Auger electrons
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

    bool                                     is_auger_enabled_;
    ParticleId                               electron_id_;
    ParticleId                               gamma_id_;
    std::unordered_map<size_type, size_type> des_to_id_;

    std::vector<AtomicRelaxElement>    host_elements_;
    std::vector<AtomicRelaxSubshell>   host_shells_;
    std::vector<AtomicRelaxTransition> host_transitions_;

    //// DEVICE DATA ////

    DeviceVector<AtomicRelaxElement>    device_elements_;
    DeviceVector<AtomicRelaxSubshell>   device_shells_;
    DeviceVector<AtomicRelaxTransition> device_transitions_;

    // HELPER FUNCTIONS
    void                      append_element(const ElementInput& inp);
    Span<AtomicRelaxSubshell> extend_shells(const ElementInput& inp);
    Span<AtomicRelaxTransition>
    extend_transitions(const std::vector<TransitionInput>& transitions);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
