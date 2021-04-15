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
#include "base/Algorithms.hh"
#include "base/DeviceVector.hh"
#include "io/ImportAtomicRelaxation.hh"
#include "physics/base/CutoffParams.hh"
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
    using MevEnergy        = units::MevEnergy;
    using SPConstCutoffs   = std::shared_ptr<const CutoffParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //@}

    struct Input
    {
        SPConstCutoffs   cutoffs;
        SPConstMaterials materials;
        SPConstParticles particles;
        bool is_auger_enabled{false}; //!< Whether to produce Auger electrons
        std::vector<ImportAtomicRelaxation> elements;
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
    void append_element(const ImportAtomicRelaxation& inp,
                        MevEnergy                     electron_cutoff,
                        MevEnergy                     gamma_cutoff);

    Span<AtomicRelaxSubshell> extend_shells(const ImportAtomicRelaxation& inp);
    Span<AtomicRelaxTransition>
    extend_transitions(const std::vector<ImportAtomicTransition>& transitions);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
