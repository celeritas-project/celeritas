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
        int       initial_shell; //!< Originating shell designator
        int       auger_shell;   //!< Auger shell designator
        real_type probability;
        real_type energy;
    };

    struct SubshellInput
    {
        int                          designator;
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
        MevEnergy electron_cut{0};    //!< Production threshold for electrons
        MevEnergy gamma_cut{0};       //!< Production threshold for photons
        bool is_auger_enabled{false}; //!< Whether to produce Auger electrons
    };

  public:
    // Construct with a vector of element identifiers
    explicit AtomicRelaxationParams(const Input& inp);

    // Access EADL data on the host
    AtomicRelaxParamsPointers host_pointers() const;

    // Access EADL data on the device
    AtomicRelaxParamsPointers device_pointers() const;

  private:
    //// HOST DATA ////

    bool                                is_auger_enabled_;
    MevEnergy                           electron_cut_;
    MevEnergy                           gamma_cut_;
    ParticleId                          electron_id_;
    ParticleId                          gamma_id_;
    std::unordered_map<int, SubshellId> des_to_id_;

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
